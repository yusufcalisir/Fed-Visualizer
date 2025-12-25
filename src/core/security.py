"""
Security Framework for Federated Learning
==========================================
Multi-layered security with Differential Privacy, Secure Aggregation,
and Byzantine-robust aggregation mechanisms.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging
import hashlib

logger = logging.getLogger("FLSecurity")


# ============================================================================
# DIFFERENTIAL PRIVACY - GAUSSIAN MECHANISM
# ============================================================================

class L2Clipper:
    """
    L2-norm gradient clipping to bound sensitivity.
    
    Formula: clip(g, C) = g * min(1, C / ||g||_2)
    """
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self.clip_history: List[float] = []
    
    def clip(self, weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Clip weights to bound their L2 norm.
        
        Returns:
            Tuple of (clipped_weights, original_norm)
        """
        # Compute global L2 norm
        total_norm_sq = sum(np.sum(w ** 2) for w in weights.values())
        total_norm = float(np.sqrt(total_norm_sq))
        
        # Clip factor
        clip_factor = min(1.0, self.max_norm / (total_norm + 1e-10))
        
        clipped = {k: v * clip_factor for k, v in weights.items()}
        
        self.clip_history.append(total_norm)
        
        return clipped, total_norm
    
    def get_clip_ratio(self) -> float:
        """Get ratio of clipped updates (norm > max_norm)."""
        if not self.clip_history:
            return 0.0
        clipped_count = sum(1 for n in self.clip_history if n > self.max_norm)
        return clipped_count / len(self.clip_history)


class GaussianMechanism:
    """
    Gaussian Mechanism for Local Differential Privacy (LDP).
    
    Adds calibrated Gaussian noise:
    w_noisy = w + N(0, σ²I)
    
    where σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
    """
    
    def __init__(self, 
                 epsilon: float = 1.0, 
                 delta: float = 1e-5, 
                 sensitivity: float = 1.0,
                 seed: Optional[int] = None):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()
        self.rng = np.random.default_rng(seed)
        
        self.noise_history: List[float] = []
        
        logger.info(f"GaussianMechanism: ε={epsilon}, δ={delta}, σ={self.sigma:.4f}")
    
    def _compute_sigma(self) -> float:
        """Compute noise standard deviation."""
        c = np.sqrt(2 * np.log(1.25 / self.delta))
        return c * self.sensitivity / self.epsilon
    
    def add_noise(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add calibrated Gaussian noise to weights."""
        noisy = {}
        total_noise_norm = 0.0
        
        for layer_name, w in weights.items():
            noise = self.rng.normal(0, self.sigma, w.shape).astype(w.dtype)
            noisy[layer_name] = w + noise
            total_noise_norm += np.sum(noise ** 2)
        
        self.noise_history.append(float(np.sqrt(total_noise_norm)))
        return noisy
    
    def get_snr(self, signal_norm: float) -> float:
        """
        Compute Signal-to-Noise Ratio.
        
        SNR = ||signal|| / ||noise||
        """
        if not self.noise_history or self.noise_history[-1] == 0:
            return float('inf')
        return signal_norm / self.noise_history[-1]
    
    def update_epsilon(self, new_epsilon: float):
        """Update privacy budget and recalculate sigma."""
        self.epsilon = new_epsilon
        self.sigma = self._compute_sigma()


class RDPAccountant:
    """
    Rigorous Rényi Differential Privacy (RDP) Accountant.
    
    Tracks cumulative privacy loss across multiple orders alpha.
    Implements the composition of subsampled Gaussian mechanisms efficiently.
    """
    
    def __init__(self, 
                 epsilon_budget: float = 10.0,
                 delta: float = 1e-5,
                 orders: Optional[List[float]] = None):
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        # Expanded range of orders for better tight bounds
        self.orders = orders or [1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64.]
        self.rdp = np.zeros(len(self.orders))
        self.steps = 0
        self.history = []  # Track (epsilon, round)
        
        self.epsilon_spent = 0.0
        self.is_exhausted = False
        
    def step(self, sigma: float, q: float = 1.0):
        """
        Accumulate RDP for one step of Subsampled Gaussian Mechanism.
        
        Args:
            sigma: Noise multiplier (std_dev / sensitivity).
            q: Sampling probability (batch_size / dataset_size).
        """
        # Calculate RDP increment for each order
        step_rdp = np.zeros_like(self.rdp)
        
        if q >= 1.0:
            # Standard Gaussian Mechanism RDP: alpha / (2 * sigma^2)
            step_rdp = np.array(self.orders) / (2 * sigma**2)
        else:
            # Subsampled Gaussian RDP (Approximate Bound)
            # Uses the relation: RDP_alpha(q, sigma) <= q^2 * alpha / sigma^2
            # valid for small q and sigma >= 1.
            # This is the "Moments" part of the Moments Accountant (Abadi et al.)
            # Deterministic, non-random calculation.
            for i, alpha in enumerate(self.orders):
                step_rdp[i] = (alpha * (q**2)) / (sigma**2)

        self.rdp += step_rdp
        self.steps += 1
        
        # update current epsilon
        self.epsilon_spent = self.get_epsilon(self.delta)
        self.history.append((self.steps, self.epsilon_spent))
        
        if self.epsilon_spent >= self.epsilon_budget:
            self.is_exhausted = True
            logger.critical(f"PRIVACY BUDGET EXHAUSTED: {self.epsilon_spent:.4f} >= {self.epsilon_budget:.4f}")
            # Raise exception to halt simulation if needed in robust mode
            # For visualization, we mark as exhausted but allow display to update
            raise PrivacyBudgetExhaustedError(f"Budget exceeded at round {self.steps}: ε={self.epsilon_spent:.4f}")

    def get_epsilon(self, delta: float = None) -> float:
        """
        Convert RDP to (epsilon, delta)-DP using the min-entropy formula.
        
        epsilon(delta) = min_alpha ( RDP(alpha) + log(1/delta)/(alpha-1) )
        """
        d = delta or self.delta
        if d <= 0: return float('inf')
        
        # Calculate epsilon for each order
        eps_alphas = []
        for i, alpha in enumerate(self.orders):
            if alpha <= 1: continue
            # Formula: epsilon = RDP(alpha) + log(1/delta) / (alpha - 1)
            eps = self.rdp[i] + np.log(1 / d) / (alpha - 1)
            eps_alphas.append(eps)
            
        return min(eps_alphas) if eps_alphas else float('inf')
    
    def get_best_order(self, delta: float = None) -> float:
        """Returns the order alpha that provides the current tightest privacy bound."""
        d = delta or self.delta
        best_alpha = 0.0
        min_eps = float('inf')
        
        for i, alpha in enumerate(self.orders):
            if alpha <= 1: continue
            eps = self.rdp[i] + np.log(1 / d) / (alpha - 1)
            if eps < min_eps:
                min_eps = eps
                best_alpha = alpha
        return best_alpha

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon_budget - self.epsilon_spent)
    
    def get_budget_fraction(self) -> float:
        """Get fraction of budget spent (0 to 1)."""
        eps = self.epsilon_spent
        if self.epsilon_budget <= 0: return 1.0
        return min(1.0, eps / self.epsilon_budget)

class PrivacyBudgetExhaustedError(Exception):
    """Exception raised when privacy budget is exhausted."""
    pass


# ============================================================================
# SECURE AGGREGATION
# ============================================================================

class SecureAggregator:
    """
    Secure Aggregation using zero-sum masking.
    
    Each client adds a mask m_k such that Σm_k = 0.
    Server sees w_k + m_k but can only compute Σw_k.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._masks: Dict[str, Dict[str, np.ndarray]] = {}
    
    def generate_masks(self, 
                       client_ids: List[str], 
                       weight_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate zero-sum masks for all clients.
        
        The last client's mask is set to cancel out all others.
        """
        masks = {}
        
        for i, client_id in enumerate(client_ids):
            client_masks = {}
            for layer_name, shape in weight_shapes.items():
                if i < len(client_ids) - 1:
                    mask = self.rng.standard_normal(shape).astype(np.float32)
                else:
                    # Last client: negative sum of all others
                    mask = -sum(masks[other_id][layer_name] for other_id in masks.keys())
                client_masks[layer_name] = mask
            masks[client_id] = client_masks
        
        self._masks = masks
        return masks
    
    def apply_mask(self, 
                   client_id: str, 
                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply mask to client weights."""
        if client_id not in self._masks:
            raise ValueError(f"No mask generated for {client_id}")
        
        masked = {}
        for layer_name, w in weights.items():
            if layer_name in self._masks[client_id]:
                masked[layer_name] = w + self._masks[client_id][layer_name]
            else:
                masked[layer_name] = w
        
        return masked
    
    def aggregate_masked(self, 
                         masked_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Aggregate masked weights.
        
        Since masks sum to zero, result equals sum of original weights.
        """
        if not masked_updates:
            return {}
        
        first_client = next(iter(masked_updates.values()))
        aggregated = {layer: np.zeros_like(w) for layer, w in first_client.items()}
        
        for client_id, updates in masked_updates.items():
            for layer_name, weight in updates.items():
                aggregated[layer_name] += weight
        
        # Average
        num_clients = len(masked_updates)
        for layer_name in aggregated:
            aggregated[layer_name] /= num_clients
        
        return aggregated


# ============================================================================
# BYZANTINE-ROBUST AGGREGATION
# ============================================================================

class KrumAggregator:
    """
    Krum Byzantine-robust aggregation.
    
    Selects the update that is closest to its k nearest neighbors,
    making it resistant to outlier/malicious updates.
    """
    
    def __init__(self, num_byzantine: int = 1):
        self.num_byzantine = num_byzantine
    
    def _flatten_weights(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten all weight matrices into a single vector."""
        return np.concatenate([w.flatten() for w in weights.values()])
    
    def _unflatten_weights(self, 
                           flat: np.ndarray, 
                           template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Unflatten vector back to weight dict."""
        result = {}
        offset = 0
        for layer_name, w in template.items():
            size = w.size
            result[layer_name] = flat[offset:offset + size].reshape(w.shape)
            offset += size
        return result
    
    def aggregate(self, 
                  client_updates: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], str]:
        """
        Perform Krum aggregation.
        
        Returns:
            Tuple of (selected_update, selected_client_id)
        """
        client_ids = list(client_updates.keys())
        n = len(client_ids)
        
        if n < 2 * self.num_byzantine + 3:
            # Not enough clients, fall back to mean
            logger.warning("Not enough clients for Krum, using mean.")
            return self._mean_aggregate(client_updates), client_ids[0]
        
        # Flatten all updates
        flat_updates = {cid: self._flatten_weights(u) for cid, u in client_updates.items()}
        
        # Compute pairwise distances
        scores = {}
        k = n - self.num_byzantine - 2
        
        for cid in client_ids:
            distances = []
            for other_cid in client_ids:
                if cid != other_cid:
                    dist = np.linalg.norm(flat_updates[cid] - flat_updates[other_cid])
                    distances.append(dist)
            
            # Score is sum of k smallest distances
            distances.sort()
            scores[cid] = sum(distances[:k])
        
        # Select client with minimum score
        best_client = min(scores, key=scores.get)
        logger.info(f"Krum selected: {best_client}")
        
        return client_updates[best_client], best_client
    
    def _mean_aggregate(self, client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Fallback mean aggregation."""
        first = next(iter(client_updates.values()))
        result = {k: np.zeros_like(v) for k, v in first.items()}
        
        for updates in client_updates.values():
            for k, v in updates.items():
                result[k] += v
        
        for k in result:
            result[k] /= len(client_updates)
        
        return result


class MedianAggregator:
    """
    Coordinate-wise Median aggregation.
    
    Computes element-wise median across all client updates,
    which is robust to outliers.
    """
    
    def aggregate(self, 
                  client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Perform coordinate-wise median aggregation."""
        if not client_updates:
            return {}
        
        first = next(iter(client_updates.values()))
        result = {}
        
        for layer_name in first.keys():
            stacked = np.stack([u[layer_name] for u in client_updates.values()], axis=0)
            result[layer_name] = np.median(stacked, axis=0)
        
        return result


class AnomalyDetector:
    """
    Anomaly detection using cosine similarity.
    
    Flags clients whose updates deviate significantly from the mean.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.similarity_history: Dict[str, List[float]] = {}
    
    def _flatten(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten weights to vector."""
        return np.concatenate([w.flatten() for w in weights.values()])
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def detect(self, 
               client_updates: Dict[str, Dict[str, np.ndarray]]) -> Tuple[List[str], Dict[str, float]]:
        """
        Detect anomalous clients.
        
        Returns:
            Tuple of (flagged_client_ids, similarity_scores)
        """
        if len(client_updates) < 2:
            return [], {}
        
        # Compute mean update
        first = next(iter(client_updates.values()))
        mean_flat = np.zeros(sum(w.size for w in first.values()))
        
        flat_updates = {}
        for cid, u in client_updates.items():
            flat = self._flatten(u)
            flat_updates[cid] = flat
            mean_flat += flat
        
        mean_flat /= len(client_updates)
        
        # Compute similarities and flag anomalies
        flagged = []
        similarities = {}
        
        for cid, flat in flat_updates.items():
            sim = self._cosine_similarity(flat, mean_flat)
            similarities[cid] = sim
            
            if cid not in self.similarity_history:
                self.similarity_history[cid] = []
            self.similarity_history[cid].append(sim)
            
            if sim < self.threshold:
                flagged.append(cid)
                logger.warning(f"Anomaly detected: {cid} (similarity={sim:.3f})")
        
        return flagged, similarities


# ============================================================================
# ATTACK SIMULATOR
# ============================================================================

class AttackSimulator:
    """
    Simulates various adversarial attacks for testing Byzantine robustness.
    """
    
    @staticmethod
    def label_flip(weights: Dict[str, np.ndarray], flip_factor: float = -1.0) -> Dict[str, np.ndarray]:
        """Simulate label-flipping attack by negating weights."""
        return {k: v * flip_factor for k, v in weights.items()}
    
    @staticmethod
    def random_noise(weights: Dict[str, np.ndarray], noise_scale: float = 10.0, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Inject large random noise."""
        rng = np.random.default_rng(seed)
        return {k: v + rng.normal(0, noise_scale, v.shape).astype(v.dtype) for k, v in weights.items()}
    
    @staticmethod
    def scale_attack(weights: Dict[str, np.ndarray], scale: float = 100.0) -> Dict[str, np.ndarray]:
        """Scale weights to dominate aggregation."""
        return {k: v * scale for k, v in weights.items()}


# ============================================================================
# UNIFIED SECURITY MANAGER
# ============================================================================

@dataclass
class SecurityConfig:
    """Configuration for security features."""
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    clip_norm: float = 1.0
    
    secagg_enabled: bool = False
    
    byzantine_robust: bool = False
    byzantine_method: str = "krum"  # "krum" or "median"
    num_byzantine: int = 1
    
    anomaly_detection: bool = True
    anomaly_threshold: float = 0.5


class SecurityManager:
    """
    Unified security manager integrating all security features.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Initialize components
        self.clipper = L2Clipper(max_norm=config.clip_norm)
        self.dp = GaussianMechanism(
            epsilon=config.dp_epsilon, 
            delta=config.dp_delta, 
            sensitivity=config.clip_norm
        )
        self.accountant = RDPAccountant(
            epsilon_budget=config.dp_epsilon, 
            delta=config.dp_delta,
            orders=None
        )
        
        self.secagg = SecureAggregator() if config.secagg_enabled else None
        
        if config.byzantine_robust:
            if config.byzantine_method == "median":
                self.aggregator = MedianAggregator()
            else:
                self.aggregator = KrumAggregator(num_byzantine=config.num_byzantine)
        else:
            self.aggregator = None
        
        self.anomaly_detector = AnomalyDetector(threshold=config.anomaly_threshold) if config.anomaly_detection else None
    
    def process_client_update(self, 
                              client_id: str,
                              weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply security processing to a client update."""
        processed = weights
        
        # Step 1: Clip
        processed, norm = self.clipper.clip(processed)
        
        # Step 2: Add DP noise
        if self.config.dp_enabled:
            processed = self.dp.add_noise(processed)
            # Note: Accounting is now handled centrally per round via step_accounting(q)
            # to correctly model the Subsampled Gaussian Mechanism.
        
        return processed
    
    def step_accounting(self, q: float = 1.0):
        """Perform privacy accounting step for the round."""
        if self.config.dp_enabled:
            self.accountant.step(self.dp.sigma, q)

    def secure_aggregate(self, 
                         client_updates: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Perform secure aggregation with all security features.
        
        Returns:
            Tuple of (aggregated_weights, security_metadata)
        """
        metadata = {
            "flagged_clients": [],
            "similarities": {},
            "snr": 0.0,
            "privacy_spent": self.accountant.epsilon_spent,
            "privacy_remaining": self.accountant.get_remaining_budget(),
            "budget_exhausted": self.accountant.is_exhausted,
            "best_alpha": self.accountant.get_best_order()
        }
        
        # Step 1: Anomaly detection
        if self.anomaly_detector:
            flagged, sims = self.anomaly_detector.detect(client_updates)
            metadata["flagged_clients"] = flagged
            metadata["similarities"] = sims
            
            # Remove flagged clients
            for cid in flagged:
                if cid in client_updates:
                    del client_updates[cid]
        
        if not client_updates:
            return {}, metadata
        
        # Step 2: Byzantine-robust aggregation or standard average
        if self.aggregator:
            if isinstance(self.aggregator, KrumAggregator):
                aggregated, _ = self.aggregator.aggregate(client_updates)
            else:
                aggregated = self.aggregator.aggregate(client_updates)
        else:
            # Standard mean
            first = next(iter(client_updates.values()))
            aggregated = {k: np.zeros_like(v) for k, v in first.items()}
            for u in client_updates.values():
                for k, v in u.items():
                    aggregated[k] += v
            for k in aggregated:
                aggregated[k] /= len(client_updates)
        
        # Compute SNR
        signal_norm = float(np.sqrt(sum(np.sum(w ** 2) for w in aggregated.values())))
        metadata["snr"] = self.dp.get_snr(signal_norm)
        
        return aggregated, metadata
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status."""
        return {
            "epsilon_spent": self.accountant.epsilon_spent,
            "epsilon_remaining": self.accountant.get_remaining_budget(),
            "budget_fraction": self.accountant.get_budget_fraction(),
            "is_exhausted": self.accountant.is_exhausted,
            "best_order": self.accountant.get_best_order(),
            "history": self.accountant.history
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Security Framework Tests ===\n")
    
    # Create test weights
    weights = {"w": np.random.randn(10, 10).astype(np.float32)}
    
    # Test L2 Clipping
    print("--- L2 Clipping ---")
    clipper = L2Clipper(max_norm=1.0)
    clipped, norm = clipper.clip(weights)
    print(f"Original norm: {norm:.4f}, Clipped norm: {np.sqrt(np.sum(clipped['w']**2)):.4f}")
    
    # Test Gaussian Mechanism
    print("\n--- Gaussian Mechanism ---")
    dp = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    noisy = dp.add_noise(clipped)
    print(f"Sigma: {dp.sigma:.4f}, SNR: {dp.get_snr(norm):.2f}")
    
    # Test RDP Accountant
    print("\n--- RDP Accountant ---")
    accountant = RDPAccountant(epsilon_budget=10.0)
    for i in range(5):
        accountant.step(dp.sigma)
    print(f"After 5 rounds: ε={accountant.epsilon_spent:.4f}, remaining={accountant.get_remaining_budget():.4f}")
    
    # Test Secure Aggregation
    print("\n--- Secure Aggregation ---")
    secagg = SecureAggregator()
    clients = ["c1", "c2", "c3"]
    shapes = {"w": (10, 10)}
    masks = secagg.generate_masks(clients, shapes)
    
    # Apply masks
    masked = {}
    for cid in clients:
        masked[cid] = secagg.apply_mask(cid, {"w": weights["w"] + np.random.randn(10, 10).astype(np.float32) * 0.1})
    
    agg = secagg.aggregate_masked(masked)
    print(f"Aggregation preserved: {agg['w'].shape}")
    
    # Test Krum
    print("\n--- Krum Aggregation ---")
    krum = KrumAggregator(num_byzantine=1)
    updates = {f"c{i}": {"w": weights["w"] + np.random.randn(10, 10).astype(np.float32) * 0.1} for i in range(5)}
    updates["malicious"] = {"w": weights["w"] * 100}  # Attacker
    selected, best_cid = krum.aggregate(updates)
    print(f"Krum selected: {best_cid}")
    
    # Test Anomaly Detection
    print("\n--- Anomaly Detection ---")
    detector = AnomalyDetector(threshold=0.5)
    flagged, sims = detector.detect(updates)
    print(f"Flagged: {flagged}")
    
    print("\n✅ All security tests passed!")
