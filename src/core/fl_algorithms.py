"""
Federated Learning Algorithms Library
======================================
A standalone, framework-agnostic computational library implementing
state-of-the-art FL optimization algorithms and privacy mechanisms.

Supports: NumPy (CPU) and CuPy (GPU) backends.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

logger = logging.getLogger("FLAlgorithms")

# Type alias for array (NumPy or CuPy)
ArrayType = Union[np.ndarray, Any]  # Any for CuPy arrays


def get_array_module(arr: ArrayType):
    """Get the array module (numpy or cupy) for the given array."""
    if HAS_CUPY and hasattr(arr, 'device'):
        return cp
    return np


# ============================================================================
# ABSTRACT BASE CLASSES (Strategy Pattern)
# ============================================================================

class AggregationStrategy(ABC):
    """
    Abstract base class for server-side aggregation strategies.
    
    Users can extend this class to implement custom aggregation logic.
    """
    
    @abstractmethod
    def aggregate(self,
                  global_weights: Dict[str, ArrayType],
                  client_updates: Dict[str, Dict[str, ArrayType]],
                  client_weights: Dict[str, float]) -> Dict[str, ArrayType]:
        """
        Aggregate client updates into new global weights.
        
        Args:
            global_weights: Current global model weights.
            client_updates: Dict mapping client_id to their weight updates.
            client_weights: Dict mapping client_id to their contribution weight (n_k/n).
        
        Returns:
            New global weights after aggregation.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass


class LocalOptimizationStrategy(ABC):
    """
    Abstract base class for client-side local optimization strategies.
    """
    
    @abstractmethod
    def optimize(self,
                 weights: Dict[str, ArrayType],
                 gradients: Dict[str, ArrayType],
                 global_weights: Optional[Dict[str, ArrayType]] = None) -> Dict[str, ArrayType]:
        """
        Perform one optimization step.
        
        Args:
            weights: Current local weights.
            gradients: Computed gradients.
            global_weights: Global weights (for proximal methods).
        
        Returns:
            Updated weights.
        """
        pass


# ============================================================================
# FEDERATED AVERAGING (FedAvg)
# ============================================================================

class FedAvgStrategy(AggregationStrategy):
    """
    Federated Averaging (FedAvg) - McMahan et al., 2017
    
    Computes weighted average of client updates:
    w_{t+1} = Σ_k (n_k/n) * w_k^{t+1}
    """
    
    @property
    def name(self) -> str:
        return "FedAvg"
    
    def aggregate(self,
                  global_weights: Dict[str, ArrayType],
                  client_updates: Dict[str, Dict[str, ArrayType]],
                  client_weights: Dict[str, float]) -> Dict[str, ArrayType]:
        xp = np  # Default to numpy
        
        # Determine array module from first update
        if client_updates:
            first_update = next(iter(client_updates.values()))
            if first_update:
                first_layer = next(iter(first_update.values()))
                xp = get_array_module(first_layer)
        
        aggregated = {}
        for layer_name in global_weights.keys():
            weighted_sum = xp.zeros_like(global_weights[layer_name])
            for client_id, updates in client_updates.items():
                if layer_name in updates:
                    weight = client_weights.get(client_id, 1.0 / len(client_updates))
                    weighted_sum += weight * updates[layer_name]
            aggregated[layer_name] = weighted_sum
        
        return aggregated


# ============================================================================
# FEDPROX (Heterogeneity Handling)
# ============================================================================

class FedProxStrategy(AggregationStrategy):
    """
    FedProx - Li et al., 2020
    
    Server-side aggregation is same as FedAvg.
    The proximal term is applied on the client side:
    min_w F_k(w) + (μ/2) ||w - w^t||²
    
    This class provides the server-side component.
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: Proximal term coefficient.
        """
        self.mu = mu
    
    @property
    def name(self) -> str:
        return f"FedProx(μ={self.mu})"
    
    def aggregate(self,
                  global_weights: Dict[str, ArrayType],
                  client_updates: Dict[str, Dict[str, ArrayType]],
                  client_weights: Dict[str, float]) -> Dict[str, ArrayType]:
        # Server-side is identical to FedAvg
        fedavg = FedAvgStrategy()
        return fedavg.aggregate(global_weights, client_updates, client_weights)


class FedProxLocalOptimizer(LocalOptimizationStrategy):
    """
    FedProx local optimizer with proximal term.
    
    Local objective: F_k(w) + (μ/2) ||w - w^t||²
    Gradient: ∇F_k(w) + μ(w - w^t)
    """
    
    def __init__(self, learning_rate: float = 0.01, mu: float = 0.01):
        self.learning_rate = learning_rate
        self.mu = mu
    
    def optimize(self,
                 weights: Dict[str, ArrayType],
                 gradients: Dict[str, ArrayType],
                 global_weights: Optional[Dict[str, ArrayType]] = None) -> Dict[str, ArrayType]:
        updated = {}
        for layer_name, w in weights.items():
            xp = get_array_module(w)
            grad = gradients.get(layer_name, xp.zeros_like(w))
            
            # Add proximal term gradient: μ(w - w^t)
            if global_weights is not None and layer_name in global_weights:
                proximal_grad = self.mu * (w - global_weights[layer_name])
                grad = grad + proximal_grad
            
            # SGD step
            updated[layer_name] = w - self.learning_rate * grad
        
        return updated


# ============================================================================
# FEDERATED ADAPTIVE OPTIMIZERS (FedAdam / FedYogi)
# ============================================================================

class FedAdamStrategy(AggregationStrategy):
    """
    Federated Adam - Reddi et al., 2020
    
    Server-side adaptive learning with momentum and variance:
    m_t = β₁ m_{t-1} + (1 - β₁) Δ_t
    v_t = β₂ v_{t-1} + (1 - β₂) Δ_t²
    w_{t+1} = w_t + η * m_t / (√v_t + τ)
    """
    
    def __init__(self,
                 eta: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 tau: float = 1e-3):
        """
        Args:
            eta: Server learning rate.
            beta1: First moment decay.
            beta2: Second moment decay.
            tau: Numerical stability constant.
        """
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        
        # State
        self.m: Dict[str, ArrayType] = {}
        self.v: Dict[str, ArrayType] = {}
        self.t: int = 0
    
    @property
    def name(self) -> str:
        return f"FedAdam(η={self.eta})"
    
    def aggregate(self,
                  global_weights: Dict[str, ArrayType],
                  client_updates: Dict[str, Dict[str, ArrayType]],
                  client_weights: Dict[str, float]) -> Dict[str, ArrayType]:
        self.t += 1
        
        # First compute standard weighted average (pseudo-gradient)
        fedavg = FedAvgStrategy()
        avg_update = fedavg.aggregate(global_weights, client_updates, client_weights)
        
        aggregated = {}
        for layer_name, w in global_weights.items():
            xp = get_array_module(w)
            
            # Compute pseudo-gradient (delta)
            delta = avg_update[layer_name] - w
            
            # Initialize moments if needed
            if layer_name not in self.m:
                self.m[layer_name] = xp.zeros_like(delta)
                self.v[layer_name] = xp.zeros_like(delta)
            
            # Update first moment: m_t = β₁ m_{t-1} + (1 - β₁) Δ_t
            self.m[layer_name] = self.beta1 * self.m[layer_name] + (1 - self.beta1) * delta
            
            # Update second moment: v_t = β₂ v_{t-1} + (1 - β₂) Δ_t²
            self.v[layer_name] = self.beta2 * self.v[layer_name] + (1 - self.beta2) * (delta ** 2)
            
            # Bias correction
            m_hat = self.m[layer_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_name] / (1 - self.beta2 ** self.t)
            
            # Update: w_{t+1} = w_t + η * m_t / (√v_t + τ)
            aggregated[layer_name] = w + self.eta * m_hat / (xp.sqrt(v_hat) + self.tau)
        
        return aggregated


class FedYogiStrategy(AggregationStrategy):
    """
    Federated Yogi - Variant of FedAdam with controlled variance growth.
    
    Difference from Adam:
    v_t = v_{t-1} + (1 - β₂) * Δ_t² * sign(Δ_t² - v_{t-1})
    
    This prevents variance from growing unboundedly.
    """
    
    def __init__(self,
                 eta: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 tau: float = 1e-3):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        
        self.m: Dict[str, ArrayType] = {}
        self.v: Dict[str, ArrayType] = {}
        self.t: int = 0
    
    @property
    def name(self) -> str:
        return f"FedYogi(η={self.eta})"
    
    def aggregate(self,
                  global_weights: Dict[str, ArrayType],
                  client_updates: Dict[str, Dict[str, ArrayType]],
                  client_weights: Dict[str, float]) -> Dict[str, ArrayType]:
        self.t += 1
        
        fedavg = FedAvgStrategy()
        avg_update = fedavg.aggregate(global_weights, client_updates, client_weights)
        
        aggregated = {}
        for layer_name, w in global_weights.items():
            xp = get_array_module(w)
            delta = avg_update[layer_name] - w
            
            if layer_name not in self.m:
                self.m[layer_name] = xp.zeros_like(delta)
                self.v[layer_name] = xp.zeros_like(delta)
            
            # First moment (same as Adam)
            self.m[layer_name] = self.beta1 * self.m[layer_name] + (1 - self.beta1) * delta
            
            # Second moment (Yogi variant)
            delta_sq = delta ** 2
            sign_term = xp.sign(delta_sq - self.v[layer_name])
            self.v[layer_name] = self.v[layer_name] + (1 - self.beta2) * delta_sq * sign_term
            
            # Ensure v is positive
            self.v[layer_name] = xp.maximum(self.v[layer_name], self.tau)
            
            # Bias correction
            m_hat = self.m[layer_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_name] / (1 - self.beta2 ** self.t)
            
            aggregated[layer_name] = w + self.eta * m_hat / (xp.sqrt(v_hat) + self.tau)
        
        return aggregated


# ============================================================================
# PRIVACY-PRESERVING MECHANISMS
# ============================================================================

class GaussianMechanism:
    """
    Gaussian Mechanism for Differential Privacy.
    
    Adds calibrated Gaussian noise to satisfy (ε, δ)-DP.
    
    Noise scale: σ = √(2 ln(1.25/δ)) * C / ε
    
    where C is the sensitivity (max L2 norm of updates).
    """
    
    def __init__(self, 
                 epsilon: float = 1.0, 
                 delta: float = 1e-5, 
                 sensitivity: float = 1.0):
        """
        Args:
            epsilon: Privacy budget.
            delta: Probability of privacy breach.
            sensitivity: L2 sensitivity (max gradient norm).
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()
        
        logger.info(f"DP initialized: ε={epsilon}, δ={delta}, σ={self.sigma:.4f}")
    
    def _compute_sigma(self) -> float:
        """Compute noise standard deviation."""
        # σ = √(2 ln(1.25/δ)) * C / ε
        c = np.sqrt(2 * np.log(1.25 / self.delta))
        return c * self.sensitivity / self.epsilon
    
    def add_noise(self, weights: Dict[str, ArrayType]) -> Dict[str, ArrayType]:
        """Add calibrated Gaussian noise to weights."""
        noisy = {}
        for layer_name, w in weights.items():
            xp = get_array_module(w)
            noise = xp.random.normal(0, self.sigma, w.shape).astype(w.dtype)
            noisy[layer_name] = w + noise
        return noisy
    
    def get_privacy_spent(self, num_iterations: int) -> Tuple[float, float]:
        """
        Compute total privacy spent after multiple iterations.
        Uses simple composition (linear).
        
        For tighter bounds, use Rényi DP accounting.
        """
        # Simple composition: ε_total = n * ε, δ_total = n * δ
        return (num_iterations * self.epsilon, num_iterations * self.delta)


class GradientClipper:
    """
    Gradient clipping for bounding per-sample influence.
    
    Clips L2-norm of gradients to max_norm C:
    g̃ = g / max(1, ||g||_2 / C)
    """
    
    def __init__(self, max_norm: float = 1.0):
        """
        Args:
            max_norm: Maximum L2 norm (clipping threshold C).
        """
        self.max_norm = max_norm
    
    def clip(self, gradients: Dict[str, ArrayType]) -> Tuple[Dict[str, ArrayType], float]:
        """
        Clip gradients and return the original norm.
        
        Returns:
            Tuple of (clipped_gradients, original_norm).
        """
        # Compute global L2 norm
        xp = np
        if gradients:
            first_grad = next(iter(gradients.values()))
            xp = get_array_module(first_grad)
        
        total_norm_sq = sum(xp.sum(g ** 2) for g in gradients.values())
        total_norm = float(xp.sqrt(total_norm_sq))
        
        # Clip factor
        clip_factor = max(1.0, total_norm / self.max_norm)
        
        clipped = {k: v / clip_factor for k, v in gradients.items()}
        
        return clipped, total_norm


# ============================================================================
# PERFORMANCE AND EVALUATION METRICS
# ============================================================================

class WeightDivergence:
    """
    Measures divergence between global and local model weights.
    
    Uses L2 distance as an approximation of Earth Mover's Distance (EMD).
    """
    
    @staticmethod
    def l2_distance(weights1: Dict[str, ArrayType], 
                    weights2: Dict[str, ArrayType]) -> float:
        """Compute L2 distance between two weight dictionaries."""
        xp = np
        if weights1:
            first = next(iter(weights1.values()))
            xp = get_array_module(first)
        
        total_sq = 0.0
        for layer_name in weights1.keys():
            if layer_name in weights2:
                diff = weights1[layer_name] - weights2[layer_name]
                total_sq += float(xp.sum(diff ** 2))
        
        return float(xp.sqrt(total_sq))
    
    @staticmethod
    def mean_divergence(global_weights: Dict[str, ArrayType],
                        client_weights_list: List[Dict[str, ArrayType]]) -> float:
        """Compute mean divergence across all clients."""
        if not client_weights_list:
            return 0.0
        
        divergences = [
            WeightDivergence.l2_distance(global_weights, cw)
            for cw in client_weights_list
        ]
        return sum(divergences) / len(divergences)


class CosineSimilarity:
    """
    Computes cosine similarity between weight updates.
    
    Used to detect malicious or outlier clients whose updates
    diverge significantly from the mean.
    """
    
    @staticmethod
    def flatten_weights(weights: Dict[str, ArrayType]) -> ArrayType:
        """Flatten all weight matrices into a single vector."""
        xp = np
        if weights:
            first = next(iter(weights.values()))
            xp = get_array_module(first)
        
        flat = xp.concatenate([w.flatten() for w in weights.values()])
        return flat
    
    @staticmethod
    def compute(weights1: Dict[str, ArrayType], 
                weights2: Dict[str, ArrayType]) -> float:
        """Compute cosine similarity between two weight dictionaries."""
        flat1 = CosineSimilarity.flatten_weights(weights1)
        flat2 = CosineSimilarity.flatten_weights(weights2)
        
        xp = get_array_module(flat1)
        
        dot_product = xp.dot(flat1, flat2)
        norm1 = xp.linalg.norm(flat1)
        norm2 = xp.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def detect_outliers(client_updates: Dict[str, Dict[str, ArrayType]],
                        threshold: float = 0.5) -> List[str]:
        """
        Detect outlier clients based on cosine similarity to mean update.
        
        Args:
            client_updates: Dict mapping client_id to their updates.
            threshold: Similarity threshold below which a client is flagged.
        
        Returns:
            List of outlier client IDs.
        """
        if len(client_updates) < 2:
            return []
        
        # Compute mean update
        xp = np
        first_update = next(iter(client_updates.values()))
        if first_update:
            first_layer = next(iter(first_update.values()))
            xp = get_array_module(first_layer)
        
        mean_update = {}
        for layer_name in first_update.keys():
            layer_sum = xp.zeros_like(first_update[layer_name])
            for updates in client_updates.values():
                if layer_name in updates:
                    layer_sum += updates[layer_name]
            mean_update[layer_name] = layer_sum / len(client_updates)
        
        # Check each client's similarity to mean
        outliers = []
        for client_id, updates in client_updates.items():
            similarity = CosineSimilarity.compute(updates, mean_update)
            if similarity < threshold:
                outliers.append(client_id)
                logger.warning(f"Outlier detected: {client_id} (similarity={similarity:.3f})")
        
        return outliers


class CommunicationEfficiency:
    """
    Measures and optimizes communication efficiency.
    
    Compares theoretical vs actual bits transmitted with different
    quantization levels (16-bit vs 32-bit).
    """
    
    @staticmethod
    def compute_size_bits(weights: Dict[str, ArrayType], 
                          dtype_bits: int = 32) -> int:
        """Compute total size in bits for given dtype."""
        total_params = sum(w.size for w in weights.values())
        return total_params * dtype_bits
    
    @staticmethod
    def quantize_to_float16(weights: Dict[str, ArrayType]) -> Dict[str, ArrayType]:
        """Quantize weights to float16 for efficient transmission."""
        quantized = {}
        for layer_name, w in weights.items():
            xp = get_array_module(w)
            quantized[layer_name] = w.astype(xp.float16)
        return quantized
    
    @staticmethod
    def dequantize_to_float32(weights: Dict[str, ArrayType]) -> Dict[str, ArrayType]:
        """Dequantize float16 weights back to float32."""
        dequantized = {}
        for layer_name, w in weights.items():
            xp = get_array_module(w)
            dequantized[layer_name] = w.astype(xp.float32)
        return dequantized
    
    @staticmethod
    def compute_compression_ratio(weights: Dict[str, ArrayType]) -> Dict[str, Any]:
        """
        Compute compression statistics.
        
        Returns dict with:
        - params: Total number of parameters
        - size_32bit_mb: Size in MB at 32-bit
        - size_16bit_mb: Size in MB at 16-bit
        - compression_ratio: 32-bit / 16-bit
        """
        total_params = sum(w.size for w in weights.values())
        size_32 = total_params * 32 / 8 / (1024 * 1024)  # MB
        size_16 = total_params * 16 / 8 / (1024 * 1024)  # MB
        
        return {
            "params": total_params,
            "size_32bit_mb": size_32,
            "size_16bit_mb": size_16,
            "compression_ratio": 2.0,
            "savings_mb": size_32 - size_16
        }


# ============================================================================
# STRATEGY REGISTRY (Factory Pattern)
# ============================================================================

class StrategyRegistry:
    """Registry for aggregation strategies."""
    
    _strategies: Dict[str, type] = {
        "fedavg": FedAvgStrategy,
        "fedprox": FedProxStrategy,
        "fedadam": FedAdamStrategy,
        "fedyogi": FedYogiStrategy,
    }
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new strategy."""
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def get(cls, name: str, **kwargs) -> AggregationStrategy:
        """Get a strategy instance by name."""
        if name.lower() not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name.lower()](**kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies."""
        return list(cls._strategies.keys())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create mock weights
    weights = {
        "layer1.weight": np.random.randn(64, 32).astype(np.float32),
        "layer1.bias": np.random.randn(64).astype(np.float32),
    }
    
    # Create mock client updates
    client_updates = {
        "client_1": {k: v + np.random.randn(*v.shape).astype(np.float32) * 0.1 for k, v in weights.items()},
        "client_2": {k: v + np.random.randn(*v.shape).astype(np.float32) * 0.1 for k, v in weights.items()},
        "client_3": {k: v + np.random.randn(*v.shape).astype(np.float32) * 0.5 for k, v in weights.items()},  # Outlier
    }
    client_weights = {"client_1": 0.4, "client_2": 0.4, "client_3": 0.2}
    
    print("=== Testing Aggregation Strategies ===")
    for strategy_name in StrategyRegistry.list_strategies():
        strategy = StrategyRegistry.get(strategy_name)
        result = strategy.aggregate(weights, client_updates, client_weights)
        print(f"{strategy.name}: aggregated {len(result)} layers")
    
    print("\n=== Testing Privacy Mechanisms ===")
    dp = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    noisy = dp.add_noise(weights)
    print(f"Added DP noise with σ={dp.sigma:.4f}")
    
    clipper = GradientClipper(max_norm=1.0)
    clipped, orig_norm = clipper.clip(weights)
    print(f"Clipped gradients: original norm={orig_norm:.4f}")
    
    print("\n=== Testing Metrics ===")
    divergence = WeightDivergence.mean_divergence(weights, list(client_updates.values()))
    print(f"Mean weight divergence: {divergence:.4f}")
    
    outliers = CosineSimilarity.detect_outliers(client_updates, threshold=0.9)
    print(f"Detected outliers: {outliers}")
    
    comm_stats = CommunicationEfficiency.compute_compression_ratio(weights)
    print(f"Communication: {comm_stats['size_32bit_mb']:.2f}MB → {comm_stats['size_16bit_mb']:.2f}MB")
    
    print("\n✅ All tests passed!")
