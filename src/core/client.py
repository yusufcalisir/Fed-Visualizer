"""
Client Simulator for Federated Learning
========================================
Implements scalable virtual client nodes with local SGD optimization,
gradient clipping, FedProx support, Non-IID data, and hardware simulation.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger("FedClient")


# ============================================================================
# HARDWARE PROFILES
# ============================================================================

class HardwareProfile(Enum):
    """Hardware profiles that dictate training speed and resource constraints."""
    HIGH_END = "high_end"           # Fast GPU, ~0.1s per epoch
    MID_RANGE = "mid_range"         # Average device, ~0.5s per epoch
    BATTERY_CONSTRAINED = "battery" # Mobile/IoT, ~1.0s per epoch


@dataclass
class HardwareStats:
    """Simulated hardware statistics for a client."""
    profile: HardwareProfile
    epoch_time: float  # Seconds per epoch
    battery_level: float = 100.0  # Percentage
    cpu_usage: float = 0.0  # Percentage
    memory_usage: float = 0.0  # MB
    
    @classmethod
    def from_profile(cls, profile: HardwareProfile) -> "HardwareStats":
        """Create hardware stats from a profile."""
        epoch_times = {
            HardwareProfile.HIGH_END: 0.1,
            HardwareProfile.MID_RANGE: 0.5,
            HardwareProfile.BATTERY_CONSTRAINED: 1.0
        }
        return cls(
            profile=profile,
            epoch_time=epoch_times[profile],
            battery_level=100.0 if profile != HardwareProfile.BATTERY_CONSTRAINED else 80.0
        )


# ============================================================================
# NON-IID DATA DISTRIBUTION
# ============================================================================

class DataDistributor:
    """
    Generates Non-IID data distributions using Dirichlet sampling.
    
    Lower alpha = more heterogeneous (label skew).
    Higher alpha = more homogeneous (IID-like).
    """
    
    def __init__(self, num_classes: int = 10, seed: Optional[int] = None):
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
    
    def dirichlet_partition(self, 
                            num_clients: int, 
                            alpha: float = 0.5,
                            total_samples: int = 60000) -> Dict[int, Dict[int, int]]:
        """
        Partition data across clients using Dirichlet distribution.
        
        Args:
            num_clients: Number of clients to partition data for.
            alpha: Concentration parameter (lower = more Non-IID).
            total_samples: Total dataset size.
        
        Returns:
            Dict mapping client_id to {class_label: num_samples}.
        """
        # Generate Dirichlet proportions for each class
        class_proportions = self.rng.dirichlet([alpha] * num_clients, size=self.num_classes)
        
        samples_per_class = total_samples // self.num_classes
        
        distributions = {i: {} for i in range(num_clients)}
        
        for class_idx in range(self.num_classes):
            # Assign samples to clients based on Dirichlet proportions
            client_samples = (class_proportions[class_idx] * samples_per_class).astype(int)
            # Handle rounding errors
            client_samples[-1] += samples_per_class - client_samples.sum()
            
            for client_id, num_samples in enumerate(client_samples):
                if num_samples > 0:
                    distributions[client_id][class_idx] = int(num_samples)
        
        return distributions
    
    def generate_class_histogram(self, distribution: Dict[int, int]) -> np.ndarray:
        """Generate a histogram of class distributions for visualization."""
        histogram = np.zeros(self.num_classes)
        for class_idx, count in distribution.items():
            histogram[class_idx] = count
        return histogram


# ============================================================================
# LOCAL OPTIMIZER
# ============================================================================

class LocalOptimizer:
    """
    Local SGD optimizer with gradient clipping and FedProx support.
    
    Update rule: w_k^{t,e+1} = w_k^{t,e} - η ∇F_k(w_k^{t,e})
    
    With FedProx: F_k(w) + (μ/2) ||w - w^t||²
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 clip_norm: Optional[float] = None,
                 fedprox_mu: float = 0.0):
        """
        Initialize the local optimizer.
        
        Args:
            learning_rate: SGD learning rate η.
            clip_norm: Max L2 norm for gradient clipping C.
            fedprox_mu: FedProx proximal term coefficient μ.
        """
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.fedprox_mu = fedprox_mu
    
    def compute_gradient(self, 
                         weights: Dict[str, np.ndarray],
                         local_data: np.ndarray,
                         local_labels: np.ndarray,
                         loss_fn: str = "cross_entropy") -> Dict[str, np.ndarray]:
        """
        Compute gradients for local training (simulated).
        
        In a real implementation, this would use automatic differentiation.
        Here we simulate gradients for demonstration.
        """
        gradients = {}
        for layer_name, weight in weights.items():
            # Simulate gradient as random noise scaled by data size
            grad = self.rng.standard_normal(weight.shape).astype(np.float32)
            grad *= 0.01  # Scale factor
            gradients[layer_name] = grad
        return gradients
    
    def clip_gradients(self, 
                       gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply gradient clipping.
        
        Formula: ∇F̃_k(w) = ∇F_k(w) / max(1, ||∇F_k(w)||_2 / C)
        """
        if self.clip_norm is None:
            return gradients
        
        # Compute global L2 norm
        total_norm_sq = sum(np.sum(g ** 2) for g in gradients.values())
        total_norm = np.sqrt(total_norm_sq)
        
        # Clip if necessary
        clip_factor = max(1.0, total_norm / self.clip_norm)
        
        return {k: v / clip_factor for k, v in gradients.items()}
    
    def apply_fedprox_term(self,
                           gradients: Dict[str, np.ndarray],
                           local_weights: Dict[str, np.ndarray],
                           global_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add FedProx proximal term to gradients.
        
        Proximal term: (μ/2) ||w - w^t||²
        Gradient contribution: μ(w - w^t)
        """
        if self.fedprox_mu == 0.0:
            return gradients
        
        modified = {}
        for layer_name, grad in gradients.items():
            if layer_name in global_weights:
                proximal_grad = self.fedprox_mu * (local_weights[layer_name] - global_weights[layer_name])
                modified[layer_name] = grad + proximal_grad
            else:
                modified[layer_name] = grad
        
        return modified
    
    def step(self,
             weights: Dict[str, np.ndarray],
             gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one SGD step.
        
        Formula: w = w - η∇F(w)
        """
        updated = {}
        for layer_name, weight in weights.items():
            if layer_name in gradients:
                updated[layer_name] = weight - self.learning_rate * gradients[layer_name]
            else:
                updated[layer_name] = weight
        return updated
    
    # Random generator for gradient simulation
    rng = np.random.default_rng()


# ============================================================================
# CLIENT NODE
# ============================================================================

@dataclass
class ClientMetrics:
    """Metrics collected during local training."""
    epoch_losses: List[float] = field(default_factory=list)
    training_time: float = 0.0
    communication_time: float = 0.0
    data_distribution: Optional[np.ndarray] = None


class ClientNode:
    """
    A simulated client node in the Federated Learning system.
    
    Manages local state, performs local training, and communicates with server.
    """
    
    def __init__(self,
                 client_id: str,
                 num_samples: int,
                 num_classes: int = 10,
                 hardware_profile: HardwareProfile = HardwareProfile.MID_RANGE,
                 learning_rate: float = 0.01,
                 clip_norm: Optional[float] = 1.0,
                 fedprox_mu: float = 0.0,
                 seed: Optional[int] = None):
        """
        Initialize a client node.
        
        Args:
            client_id: Unique identifier for this client.
            num_samples: Number of local training samples.
            num_classes: Number of classes in the classification task.
            hardware_profile: Hardware profile for simulation.
            learning_rate: Local SGD learning rate.
            clip_norm: Gradient clipping norm (None to disable).
            fedprox_mu: FedProx proximal term coefficient (0 to disable).
            seed: Random seed for reproducibility.
        """
        self.client_id = client_id
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Hardware simulation
        self.hardware = HardwareStats.from_profile(hardware_profile)
        
        # Optimizer
        self.optimizer = LocalOptimizer(
            learning_rate=learning_rate,
            clip_norm=clip_norm,
            fedprox_mu=fedprox_mu
        )
        
        # Local state
        self.local_weights: Optional[Dict[str, np.ndarray]] = None
        self.global_weights_cache: Optional[Dict[str, np.ndarray]] = None
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # Metrics
        self.metrics = ClientMetrics()
        
        # Data distribution (Non-IID)
        self._generate_data_distribution()
        
        # Logs for Streamlit
        self.logs: List[Dict[str, Any]] = []
    
    def _generate_data_distribution(self):
        """Generate a Non-IID data distribution for this client."""
        # Use Dirichlet to create skewed distribution
        proportions = self.rng.dirichlet([0.5] * self.num_classes)
        self.data_distribution = (proportions * self.num_samples).astype(int)
        # Ensure sum matches num_samples
        self.data_distribution[-1] += self.num_samples - self.data_distribution.sum()
        self.metrics.data_distribution = self.data_distribution
    
    def receive_global_weights(self, weights: Dict[str, np.ndarray]):
        """Receive global weights from server and initialize local weights."""
        self.global_weights_cache = {k: v.copy() for k, v in weights.items()}
        self.local_weights = {k: v.copy() for k, v in weights.items()}
        self._log(f"Received global weights ({len(weights)} layers).")
    
    def local_train(self, 
                    epochs: int = 5,
                    batch_size: int = 32) -> Tuple[Dict[str, np.ndarray], ClientMetrics]:
        """
        Perform local training for the specified number of epochs.
        
        Args:
            epochs: Number of local epochs.
            batch_size: Batch size for training.
        
        Returns:
            Tuple of (updated weights, training metrics).
        """
        if self.local_weights is None:
            raise ValueError("Must receive global weights before training.")
        
        self.metrics.epoch_losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            # Simulate local training
            epoch_start = time.time()
            
            # Generate synthetic gradients
            gradients = {}
            for layer_name, weight in self.local_weights.items():
                # Simulate gradient as random noise + signal
                grad = self.rng.standard_normal(weight.shape).astype(np.float32) * 0.01
                gradients[layer_name] = grad
            
            # Apply gradient clipping
            gradients = self.optimizer.clip_gradients(gradients)
            
            # Apply FedProx term if enabled
            if self.optimizer.fedprox_mu > 0:
                gradients = self.optimizer.apply_fedprox_term(
                    gradients, self.local_weights, self.global_weights_cache
                )
            
            # SGD step
            self.local_weights = self.optimizer.step(self.local_weights, gradients)
            
            # Simulate training time based on hardware profile
            time.sleep(self.hardware.epoch_time * 0.1)  # Scaled for demo
            
            # Compute simulated loss (decreasing over epochs)
            base_loss = 2.0 - (epoch * 0.3) + self.rng.uniform(-0.1, 0.1)
            loss = max(0.1, base_loss)
            self.metrics.epoch_losses.append(loss)
            
            # Update hardware stats
            self.hardware.battery_level = max(0, self.hardware.battery_level - 0.5)
            self.hardware.cpu_usage = 70 + self.rng.uniform(-10, 10)
            
            self._log(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        
        self.metrics.training_time = time.time() - start_time
        self._log(f"Local training complete in {self.metrics.training_time:.2f}s.")
        
        return self.local_weights, self.metrics
    
    def compute_weight_delta(self) -> Dict[str, np.ndarray]:
        """
        Compute weight delta for efficient transmission.
        
        Formula: Δw_k = w_k^{t+1} - w^t
        """
        if self.local_weights is None or self.global_weights_cache is None:
            raise ValueError("Both local and global weights must be set.")
        
        delta = {}
        for layer_name, local_w in self.local_weights.items():
            if layer_name in self.global_weights_cache:
                delta[layer_name] = local_w - self.global_weights_cache[layer_name]
            else:
                delta[layer_name] = local_w
        
        return delta
    
    def compute_checksum(self, weights: Dict[str, np.ndarray]) -> str:
        """
        Compute SHA-256 checksum for weight integrity verification.
        """
        hasher = hashlib.sha256()
        for layer_name in sorted(weights.keys()):
            hasher.update(layer_name.encode())
            hasher.update(weights[layer_name].tobytes())
        return hasher.hexdigest()
    
    def verify_weights(self, 
                       weights: Dict[str, np.ndarray], 
                       expected_checksum: str) -> bool:
        """Verify weight integrity using checksum."""
        actual_checksum = self.compute_checksum(weights)
        return actual_checksum == expected_checksum
    
    def simulate_network_latency(self) -> float:
        """
        Simulate network latency using Poisson distribution.
        
        Returns the delay in seconds.
        """
        # Poisson-distributed delay (mean = 0.1s for high-end, 0.5s for constrained)
        mean_delay = {
            HardwareProfile.HIGH_END: 0.05,
            HardwareProfile.MID_RANGE: 0.1,
            HardwareProfile.BATTERY_CONSTRAINED: 0.3
        }[self.hardware.profile]
        
        delay = self.rng.poisson(mean_delay * 10) / 10.0  # Scale for reasonable values
        return delay
    
    def get_streamlit_data(self) -> Dict[str, Any]:
        """
        Get data formatted for Streamlit visualization.
        """
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "hardware_profile": self.hardware.profile.value,
            "battery_level": self.hardware.battery_level,
            "cpu_usage": self.hardware.cpu_usage,
            "data_distribution": self.data_distribution.tolist() if self.data_distribution is not None else [],
            "epoch_losses": self.metrics.epoch_losses,
            "training_time": self.metrics.training_time,
            "logs": self.logs[-5:]  # Last 5 logs
        }
    
    def _log(self, message: str):
        """Add a log entry."""
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "client_id": self.client_id,
            "message": message
        }
        self.logs.append(entry)
        logger.info(f"[{self.client_id}] {message}")


# ============================================================================
# CLIENT MANAGER (For scaling to thousands of clients)
# ============================================================================

class ClientManager:
    """
    Manages a pool of simulated clients for large-scale FL experiments.
    """
    
    def __init__(self, 
                 num_clients: int = 100,
                 num_classes: int = 10,
                 dirichlet_alpha: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize the client manager.
        
        Args:
            num_clients: Total number of clients to simulate.
            num_classes: Number of classes in the dataset.
            dirichlet_alpha: Dirichlet concentration for Non-IID.
            seed: Random seed.
        """
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        
        # Generate Non-IID data distributions
        self.distributor = DataDistributor(num_classes=num_classes, seed=seed)
        self.data_distributions = self.distributor.dirichlet_partition(
            num_clients=num_clients,
            alpha=dirichlet_alpha
        )
        
        # Create clients
        self.clients: Dict[str, ClientNode] = {}
        self._create_clients()
    
    def _create_clients(self):
        """Create all client nodes with varied hardware profiles."""
        profiles = [HardwareProfile.HIGH_END, HardwareProfile.MID_RANGE, HardwareProfile.BATTERY_CONSTRAINED]
        profile_weights = [0.2, 0.5, 0.3]  # 20% high-end, 50% mid, 30% constrained
        
        for i in range(self.num_clients):
            client_id = f"client_{i+1}"
            
            # Assign hardware profile based on weights
            profile = self.rng.choice(profiles, p=profile_weights)
            
            # Calculate samples from distribution
            client_dist = self.data_distributions.get(i, {})
            num_samples = sum(client_dist.values()) if client_dist else 100
            
            client = ClientNode(
                client_id=client_id,
                num_samples=max(100, num_samples),
                num_classes=self.num_classes,
                hardware_profile=profile,
                seed=self.rng.integers(0, 10000)
            )
            
            # Override data distribution with Dirichlet-generated one
            if client_dist:
                dist_array = np.zeros(self.num_classes)
                for class_idx, count in client_dist.items():
                    dist_array[class_idx] = count
                client.data_distribution = dist_array.astype(int)
                client.metrics.data_distribution = client.data_distribution
            
            self.clients[client_id] = client
    
    def select_clients(self, 
                       k: int,
                       selection_strategy: str = "random") -> List[ClientNode]:
        """
        Select k clients for a training round.
        
        Args:
            k: Number of clients to select.
            selection_strategy: 'random', 'priority' (by samples), or 'available' (by battery).
        
        Returns:
            List of selected ClientNode instances.
        """
        client_list = list(self.clients.values())
        
        if selection_strategy == "random":
            selected = self.rng.choice(client_list, size=min(k, len(client_list)), replace=False)
        elif selection_strategy == "priority":
            # Prioritize clients with more data
            sorted_clients = sorted(client_list, key=lambda c: c.num_samples, reverse=True)
            selected = sorted_clients[:k]
        elif selection_strategy == "available":
            # Select clients with sufficient battery
            available = [c for c in client_list if c.hardware.battery_level > 20]
            selected = self.rng.choice(available, size=min(k, len(available)), replace=False)
        else:
            selected = client_list[:k]
        
        return list(selected)
    
    def get_all_streamlit_data(self) -> List[Dict[str, Any]]:
        """Get Streamlit data for all clients."""
        return [c.get_streamlit_data() for c in self.clients.values()]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a client manager with 10 clients for testing
    manager = ClientManager(num_clients=10, dirichlet_alpha=0.3, seed=42)
    
    # Select 5 clients for training
    selected = manager.select_clients(k=5)
    print(f"Selected {len(selected)} clients for training:")
    
    # Simulate one round of training
    mock_weights = {
        "layer1.weight": np.random.randn(64, 32).astype(np.float32),
        "layer1.bias": np.random.randn(64).astype(np.float32)
    }
    
    for client in selected:
        print(f"\n--- {client.client_id} ({client.hardware.profile.value}) ---")
        print(f"Data distribution: {client.data_distribution}")
        
        # Receive global weights
        client.receive_global_weights(mock_weights)
        
        # Perform local training
        local_weights, metrics = client.local_train(epochs=3)
        
        # Compute delta
        delta = client.compute_weight_delta()
        checksum = client.compute_checksum(local_weights)
        
        print(f"Training time: {metrics.training_time:.2f}s")
        print(f"Epoch losses: {metrics.epoch_losses}")
        print(f"Checksum: {checksum[:16]}...")
        print(f"Battery: {client.hardware.battery_level:.1f}%")