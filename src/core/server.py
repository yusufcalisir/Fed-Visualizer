"""
Central Aggregation Server for Federated Learning
==================================================
Implements FedAvg aggregation with weighted averaging, convergence monitoring,
and modular architecture for easy algorithm swapping.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FedServer")


class AggregationPhase(Enum):
    """Phases of the FL lifecycle."""
    IDLE = "idle"
    BROADCAST = "broadcast"
    TRAINING = "training"
    WAITING = "waiting_for_updates"
    RECEIVING = "receiving"
    AGGREGATING = "aggregating"


@dataclass
class ClientMetadata:
    """Metadata for a single client."""
    client_id: str
    num_samples: int
    is_active: bool = True
    last_seen: float = field(default_factory=time.time)
    contribution_weight: float = 0.0


@dataclass
class GlobalState:
    """
    Maintains the global state of the Federated Learning server.
    
    Attributes:
        round: Current training round number.
        global_weights: Dictionary mapping layer names to NumPy arrays.
        client_registry: Dictionary mapping client IDs to ClientMetadata.
        phase: Current phase of the FL lifecycle.
        logs: List of log entries for Streamlit integration.
    """
    round: int = 0
    global_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    client_registry: Dict[str, ClientMetadata] = field(default_factory=dict)
    phase: AggregationPhase = AggregationPhase.IDLE
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Convergence tracking
    prev_weights: Optional[Dict[str, np.ndarray]] = None
    gradient_norms: List[float] = field(default_factory=list)
    
    def add_log(self, message: str, level: str = "INFO"):
        """Add a log entry with timestamp."""
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "round": self.round,
            "phase": self.phase.value,
            "level": level,
            "message": message
        }
        self.logs.append(entry)
        if len(self.logs) > 100:
            self.logs.pop(0)  # Keep last 100 logs
        logger.info(f"[Round {self.round}] {message}")


class FederatedServer:
    """
    Central Aggregation Server for Federated Learning.
    
    Implements FedAvg with support for FedProx and FedAdam variants.
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, np.ndarray],
                 min_clients: int = 2,
                 client_timeout: float = 30.0,
                 aggregation_strategy: str = "fedavg"):
        """
        Initialize the Federated Server.
        
        Args:
            initial_weights: Initial global model weights.
            min_clients: Minimum quorum for aggregation.
            client_timeout: Timeout in seconds for straggler mitigation.
            aggregation_strategy: One of 'fedavg', 'fedprox', 'fedadam'.
        """
        self.state = GlobalState(global_weights=initial_weights.copy())
        self.min_clients = min_clients
        self.client_timeout = client_timeout
        self.aggregation_strategy = aggregation_strategy
        
        # Aggregation strategy registry
        self._strategies: Dict[str, Callable] = {
            "fedavg": self._fedavg_aggregate,
            "fedprox": self._fedprox_aggregate,
            "fedadam": self._fedadam_aggregate
        }
        
        # FedAdam specific state
        self._adam_m: Dict[str, np.ndarray] = {}  # First moment
        self._adam_v: Dict[str, np.ndarray] = {}  # Second moment
        self._adam_t: int = 0  # Timestep
        
        self.state.add_log("Server initialized.")
    
    def register_client(self, client_id: str, num_samples: int) -> None:
        """Register a new client or update existing client metadata."""
        if client_id in self.state.client_registry:
            self.state.client_registry[client_id].num_samples = num_samples
            self.state.client_registry[client_id].last_seen = time.time()
            self.state.client_registry[client_id].is_active = True
        else:
            self.state.client_registry[client_id] = ClientMetadata(
                client_id=client_id,
                num_samples=num_samples
            )
        self.state.add_log(f"Client '{client_id}' registered with {num_samples} samples.")
    
    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """Return a copy of the current global weights for broadcasting."""
        self.state.phase = AggregationPhase.BROADCAST
        self.state.add_log("Broadcasting global weights to clients.")
        return {k: v.copy() for k, v in self.state.global_weights.items()}
    
    def aggregate(self, 
                  client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates using the configured strategy.
        
        Args:
            client_updates: Dict mapping client_id to their weight updates.
        
        Returns:
            Updated global weights.
        """
        self.state.phase = AggregationPhase.AGGREGATING
        self.state.round += 1
        
        # Store previous weights for convergence monitoring
        self.state.prev_weights = {k: v.copy() for k, v in self.state.global_weights.items()}
        
        # Calculate contribution weights based on sample sizes
        total_samples = sum(
            self.state.client_registry[cid].num_samples 
            for cid in client_updates.keys()
            if cid in self.state.client_registry
        )
        
        for cid in client_updates.keys():
            if cid in self.state.client_registry:
                self.state.client_registry[cid].contribution_weight = (
                    self.state.client_registry[cid].num_samples / total_samples
                )
        
        # Log weight distribution
        weight_dist = {
            cid: self.state.client_registry[cid].contribution_weight 
            for cid in client_updates.keys() 
            if cid in self.state.client_registry
        }
        self.state.add_log(f"Client weight distribution: {weight_dist}")
        
        # Perform aggregation
        strategy_fn = self._strategies.get(self.aggregation_strategy, self._fedavg_aggregate)
        self.state.global_weights = strategy_fn(client_updates)
        
        # Compute convergence metric
        grad_norm = self._compute_gradient_norm()
        self.state.gradient_norms.append(grad_norm)
        self.state.add_log(f"Round {self.state.round} complete. Gradient L2-norm: {grad_norm:.6f}")
        
        self.state.phase = AggregationPhase.IDLE
        return self.state.global_weights
    
    def _fedavg_aggregate(self, 
                          client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        FedAvg: Weighted average of client updates.
        
        Formula: w_{t+1} = sum_{k in S_t} (n_k / n) * w_k^{t+1}
        """
        self.state.add_log("Aggregating with FedAvg...")
        
        aggregated = {}
        for layer_name in self.state.global_weights.keys():
            weighted_sum = np.zeros_like(self.state.global_weights[layer_name])
            for cid, updates in client_updates.items():
                if cid in self.state.client_registry and layer_name in updates:
                    weight = self.state.client_registry[cid].contribution_weight
                    weighted_sum += weight * updates[layer_name]
            aggregated[layer_name] = weighted_sum
        
        return aggregated
    
    def _fedprox_aggregate(self, 
                           client_updates: Dict[str, Dict[str, np.ndarray]],
                           mu: float = 0.01) -> Dict[str, np.ndarray]:
        """
        FedProx: FedAvg with proximal term (stub).
        
        The proximal term is applied on the client side during local training.
        Server-side aggregation is the same as FedAvg.
        
        Note: Full FedProx requires client-side modification to add:
              (mu/2) * ||w - w_global||^2 to the loss function.
        """
        self.state.add_log(f"Aggregating with FedProx (mu={mu})...")
        # Server-side is identical to FedAvg
        return self._fedavg_aggregate(client_updates)
    
    def _fedadam_aggregate(self,
                           client_updates: Dict[str, Dict[str, np.ndarray]],
                           beta1: float = 0.9,
                           beta2: float = 0.99,
                           tau: float = 1e-3,
                           eta: float = 0.01) -> Dict[str, np.ndarray]:
        """
        FedAdam: Adaptive federated optimization.
        
        Uses Adam-style updates on the server side.
        """
        self.state.add_log("Aggregating with FedAdam...")
        self._adam_t += 1
        
        # Compute pseudo-gradient (difference from global to aggregated)
        avg_update = self._fedavg_aggregate(client_updates)
        
        aggregated = {}
        for layer_name in self.state.global_weights.keys():
            # Pseudo-gradient
            delta = avg_update[layer_name] - self.state.global_weights[layer_name]
            
            # Initialize moments if needed
            if layer_name not in self._adam_m:
                self._adam_m[layer_name] = np.zeros_like(delta)
                self._adam_v[layer_name] = np.zeros_like(delta)
            
            # Update moments
            self._adam_m[layer_name] = beta1 * self._adam_m[layer_name] + (1 - beta1) * delta
            self._adam_v[layer_name] = beta2 * self._adam_v[layer_name] + (1 - beta2) * (delta ** 2)
            
            # Bias correction
            m_hat = self._adam_m[layer_name] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[layer_name] / (1 - beta2 ** self._adam_t)
            
            # Update
            aggregated[layer_name] = (
                self.state.global_weights[layer_name] + 
                eta * m_hat / (np.sqrt(v_hat) + tau)
            )
        
        return aggregated
    
    def _compute_gradient_norm(self) -> float:
        """
        Compute L2-norm of weight updates for convergence monitoring.
        
        Formula: ||w_{t+1} - w_t||_2
        """
        if self.state.prev_weights is None:
            return 0.0
        
        total_norm_sq = 0.0
        for layer_name in self.state.global_weights.keys():
            if layer_name in self.state.prev_weights:
                diff = self.state.global_weights[layer_name] - self.state.prev_weights[layer_name]
                total_norm_sq += np.sum(diff ** 2)
        
        return np.sqrt(total_norm_sq)
    
    def check_straggler(self, received_clients: List[str]) -> bool:
        """
        Check if minimum quorum is met for aggregation.
        
        Args:
            received_clients: List of client IDs that have sent updates.
        
        Returns:
            True if quorum is met, False otherwise.
        """
        if len(received_clients) >= self.min_clients:
            self.state.add_log(f"Quorum met: {len(received_clients)}/{self.min_clients} clients.")
            return True
        else:
            self.state.add_log(
                f"Quorum not met: {len(received_clients)}/{self.min_clients} clients.", 
                level="WARNING"
            )
            return False
    
    def get_logs_for_streamlit(self) -> List[Dict[str, Any]]:
        """Return logs formatted for Streamlit display."""
        return self.state.logs.copy()
    
    def get_client_weights_distribution(self) -> Dict[str, float]:
        """Return current client contribution weights for visualization."""
        return {
            cid: meta.contribution_weight 
            for cid, meta in self.state.client_registry.items()
        }
    
    def get_convergence_history(self) -> List[float]:
        """Return history of gradient norms for convergence plotting."""
        return self.state.gradient_norms.copy()


# Example usage and testing
if __name__ == "__main__":
    # Create mock initial weights (simulating a simple neural network)
    initial_weights = {
        "layer1.weight": np.random.randn(64, 32),
        "layer1.bias": np.random.randn(64),
        "layer2.weight": np.random.randn(10, 64),
        "layer2.bias": np.random.randn(10)
    }
    
    # Initialize server
    server = FederatedServer(
        initial_weights=initial_weights,
        min_clients=2,
        aggregation_strategy="fedavg"
    )
    
    # Register clients
    server.register_client("client_1", num_samples=1000)
    server.register_client("client_2", num_samples=500)
    server.register_client("client_3", num_samples=750)
    
    # Simulate client updates (in practice, these come from local training)
    client_updates = {
        "client_1": {
            "layer1.weight": initial_weights["layer1.weight"] + np.random.randn(64, 32) * 0.1,
            "layer1.bias": initial_weights["layer1.bias"] + np.random.randn(64) * 0.1,
            "layer2.weight": initial_weights["layer2.weight"] + np.random.randn(10, 64) * 0.1,
            "layer2.bias": initial_weights["layer2.bias"] + np.random.randn(10) * 0.1
        },
        "client_2": {
            "layer1.weight": initial_weights["layer1.weight"] + np.random.randn(64, 32) * 0.1,
            "layer1.bias": initial_weights["layer1.bias"] + np.random.randn(64) * 0.1,
            "layer2.weight": initial_weights["layer2.weight"] + np.random.randn(10, 64) * 0.1,
            "layer2.bias": initial_weights["layer2.bias"] + np.random.randn(10) * 0.1
        }
    }
    
    # Perform aggregation
    if server.check_straggler(list(client_updates.keys())):
        new_weights = server.aggregate(client_updates)
        print(f"Aggregation complete. Gradient norm: {server.state.gradient_norms[-1]:.6f}")
    
    # Print logs
    print("\n--- Server Logs ---")
    for log in server.get_logs_for_streamlit():
        print(f"[{log['timestamp']}] {log['message']}")