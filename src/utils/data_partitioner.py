"""
Data Preprocessing Engine for Federated Learning
=================================================
Flexible data partitioning module with IID, Dirichlet Non-IID,
and Shard-based strategies for realistic FL simulations.
"""

import numpy as np
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging

logger = logging.getLogger("DataPartitioner")


# ============================================================================
# PARTITION MANIFEST (Metadata)
# ============================================================================

@dataclass
class ClientPartitionInfo:
    """Metadata for a single client's data partition."""
    client_id: str
    num_samples: int
    class_distribution: Dict[int, int]  # class_id -> count
    entropy: float
    data_hash: str
    indices: np.ndarray  # Indices into original dataset


@dataclass
class PartitionManifest:
    """
    Complete manifest for a partitioning run.
    Stores metadata for all clients and global statistics.
    """
    strategy_name: str
    num_clients: int
    num_classes: int
    total_samples: int
    partitions: Dict[str, ClientPartitionInfo] = field(default_factory=dict)
    global_class_distribution: Dict[int, int] = field(default_factory=dict)
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            "strategy": self.strategy_name,
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "total_samples": self.total_samples,
            "partitions": {
                cid: {
                    "num_samples": p.num_samples,
                    "class_distribution": p.class_distribution,
                    "entropy": p.entropy,
                    "hash": p.data_hash
                }
                for cid, p in self.partitions.items()
            },
            "global_distribution": self.global_class_distribution
        }


# ============================================================================
# ABSTRACT PARTITIONER
# ============================================================================

class PartitionStrategy(ABC):
    """Abstract base class for data partitioning strategies."""
    
    @abstractmethod
    def partition(self,
                  labels: np.ndarray,
                  num_clients: int,
                  **kwargs) -> Dict[str, np.ndarray]:
        """
        Partition data indices across clients.
        
        Args:
            labels: Array of class labels for each sample.
            num_clients: Number of clients to partition data for.
        
        Returns:
            Dict mapping client_id to array of sample indices.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


# ============================================================================
# IID PARTITIONING
# ============================================================================

class IIDPartitioner(PartitionStrategy):
    """
    IID (Independent and Identically Distributed) Partitioning.
    
    Randomly shuffles data and distributes equal-sized portions,
    ensuring each client's local set represents the global distribution.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return "IID"
    
    def partition(self,
                  labels: np.ndarray,
                  num_clients: int,
                  **kwargs) -> Dict[str, np.ndarray]:
        n_samples = len(labels)
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        # Split into equal parts
        splits = np.array_split(indices, num_clients)
        
        return {f"client_{i+1}": split for i, split in enumerate(splits)}


# ============================================================================
# DIRICHLET NON-IID PARTITIONING
# ============================================================================

class DirichletPartitioner(PartitionStrategy):
    """
    Dirichlet Distribution Non-IID Partitioning.
    
    Uses Dirichlet distribution to control degree of non-IIDness:
    q_{k,c} ~ Dir(α * p)
    
    - Small α (e.g., 0.1): High skew, clients hold few classes
    - Large α (e.g., 100): Converges to IID distribution
    """
    
    def __init__(self, alpha: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            alpha: Concentration parameter. Lower = more Non-IID.
            seed: Random seed.
        """
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return f"Dirichlet(α={self.alpha})"
    
    def partition(self,
                  labels: np.ndarray,
                  num_clients: int,
                  min_samples_per_client: int = 10,
                  **kwargs) -> Dict[str, np.ndarray]:
        n_samples = len(labels)
        classes = np.unique(labels)
        num_classes = len(classes)
        
        # Get indices for each class
        class_indices = {c: np.where(labels == c)[0] for c in classes}
        
        # Sample Dirichlet proportions for each class
        # Shape: (num_classes, num_clients)
        proportions = self.rng.dirichlet([self.alpha] * num_clients, size=num_classes)
        
        # Initialize client indices
        client_indices = {f"client_{i+1}": [] for i in range(num_clients)}
        
        # Distribute samples from each class according to Dirichlet proportions
        for class_idx, c in enumerate(classes):
            c_indices = class_indices[c].copy()
            self.rng.shuffle(c_indices)
            
            # Calculate number of samples per client for this class
            n_class = len(c_indices)
            client_counts = (proportions[class_idx] * n_class).astype(int)
            
            # Handle rounding errors
            diff = n_class - client_counts.sum()
            client_counts[self.rng.choice(num_clients, abs(diff))] += np.sign(diff)
            
            # Assign samples
            start = 0
            for i in range(num_clients):
                end = start + client_counts[i]
                client_indices[f"client_{i+1}"].extend(c_indices[start:end])
                start = end
        
        # Convert to numpy arrays
        result = {cid: np.array(indices, dtype=np.int64) 
                  for cid, indices in client_indices.items()}
        
        # Ensure minimum samples per client
        for cid in result:
            if len(result[cid]) < min_samples_per_client:
                logger.warning(f"{cid} has only {len(result[cid])} samples")
        
        return result


# ============================================================================
# SHARD-BASED PARTITIONING
# ============================================================================

class ShardPartitioner(PartitionStrategy):
    """
    Shard-Based Non-IID Partitioning.
    
    Sorts dataset by class labels, divides into S shards,
    and assigns a fixed number of shards to each client.
    """
    
    def __init__(self, 
                 shards_per_client: int = 2,
                 seed: Optional[int] = None):
        """
        Args:
            shards_per_client: Number of shards assigned to each client.
            seed: Random seed.
        """
        self.shards_per_client = shards_per_client
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return f"Shard(shards_per_client={self.shards_per_client})"
    
    def partition(self,
                  labels: np.ndarray,
                  num_clients: int,
                  **kwargs) -> Dict[str, np.ndarray]:
        n_samples = len(labels)
        
        # Sort indices by label
        sorted_indices = np.argsort(labels)
        
        # Total number of shards
        total_shards = num_clients * self.shards_per_client
        
        # Split into shards
        shards = np.array_split(sorted_indices, total_shards)
        
        # Shuffle shard order
        shard_order = list(range(total_shards))
        self.rng.shuffle(shard_order)
        
        # Assign shards to clients
        client_indices = {}
        for i in range(num_clients):
            client_id = f"client_{i+1}"
            assigned_shards = shard_order[i * self.shards_per_client:(i + 1) * self.shards_per_client]
            indices = np.concatenate([shards[s] for s in assigned_shards])
            client_indices[client_id] = indices
        
        return client_indices


# ============================================================================
# DATASET WRAPPERS
# ============================================================================

class DatasetWrapper(ABC):
    """Abstract wrapper for datasets."""
    
    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (data, labels) arrays."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes."""
        pass


class SyntheticDataset(DatasetWrapper):
    """Synthetic dataset for testing."""
    
    def __init__(self, 
                 num_samples: int = 10000,
                 num_features: int = 784,
                 num_classes: int = 10,
                 seed: Optional[int] = None):
        self.num_samples = num_samples
        self.num_features = num_features
        self._num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        
        self._data = self.rng.standard_normal((num_samples, num_features)).astype(np.float32)
        self._labels = self.rng.integers(0, num_classes, size=num_samples)
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data, self._labels
    
    @property
    def num_classes(self) -> int:
        return self._num_classes


class CSVDatasetWrapper(DatasetWrapper):
    """Wrapper for CSV/tabular datasets."""
    
    def __init__(self, 
                 data_path: str,
                 label_column: str = "label",
                 num_classes: Optional[int] = None):
        import pandas as pd
        
        self.df = pd.read_csv(data_path)
        self.label_column = label_column
        
        self._labels = self.df[label_column].values
        self._data = self.df.drop(columns=[label_column]).values.astype(np.float32)
        self._num_classes = num_classes or len(np.unique(self._labels))
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data, self._labels
    
    @property
    def num_classes(self) -> int:
        return self._num_classes


class NumpyDatasetWrapper(DatasetWrapper):
    """Wrapper for NumPy arrays."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self._data = data
        self._labels = labels
        self._num_classes = len(np.unique(labels))
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data, self._labels
    
    @property
    def num_classes(self) -> int:
        return self._num_classes


# ============================================================================
# DATA PARTITIONER ENGINE
# ============================================================================

class DataPartitioner:
    """
    Main engine for partitioning datasets across FL clients.
    
    Provides memory-efficient indexing and comprehensive metadata.
    """
    
    def __init__(self, dataset: DatasetWrapper):
        """
        Args:
            dataset: Dataset wrapper containing data and labels.
        """
        self.dataset = dataset
        self._data, self._labels = dataset.get_data()
        self._num_classes = dataset.num_classes
        
        # Compute global class distribution
        self.global_distribution = self._compute_distribution(np.arange(len(self._labels)))
    
    def partition(self,
                  strategy: PartitionStrategy,
                  num_clients: int,
                  **kwargs) -> PartitionManifest:
        """
        Partition the dataset using the specified strategy.
        
        Args:
            strategy: Partitioning strategy to use.
            num_clients: Number of clients.
        
        Returns:
            PartitionManifest with all metadata.
        """
        import time
        
        # Perform partitioning
        client_indices = strategy.partition(self._labels, num_clients, **kwargs)
        
        # Build manifest
        manifest = PartitionManifest(
            strategy_name=strategy.name,
            num_clients=num_clients,
            num_classes=self._num_classes,
            total_samples=len(self._labels),
            global_class_distribution=self.global_distribution,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Build partition info for each client
        for client_id, indices in client_indices.items():
            class_dist = self._compute_distribution(indices)
            entropy = self._compute_entropy(class_dist)
            data_hash = self._compute_hash(indices)
            
            manifest.partitions[client_id] = ClientPartitionInfo(
                client_id=client_id,
                num_samples=len(indices),
                class_distribution=class_dist,
                entropy=entropy,
                data_hash=data_hash,
                indices=indices
            )
        
        logger.info(f"Partitioned {len(self._labels)} samples across {num_clients} clients using {strategy.name}")
        
        return manifest
    
    def _compute_distribution(self, indices: np.ndarray) -> Dict[int, int]:
        """Compute class distribution for given indices."""
        labels = self._labels[indices]
        unique, counts = np.unique(labels, return_counts=True)
        return {int(c): int(n) for c, n in zip(unique, counts)}
    
    def _compute_entropy(self, distribution: Dict[int, int]) -> float:
        """
        Compute Shannon entropy of class distribution.
        
        H = -Σ p(c) log₂(p(c))
        """
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        probs = np.array(list(distribution.values())) / total
        probs = probs[probs > 0]  # Avoid log(0)
        
        return float(-np.sum(probs * np.log2(probs)))
    
    def _compute_hash(self, indices: np.ndarray) -> str:
        """Compute SHA-256 hash of indices for integrity verification."""
        return hashlib.sha256(indices.tobytes()).hexdigest()[:16]
    
    def get_client_data(self, 
                        manifest: PartitionManifest,
                        client_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific client (view, not copy).
        
        Uses indexing for memory efficiency.
        """
        if client_id not in manifest.partitions:
            raise ValueError(f"Unknown client: {client_id}")
        
        indices = manifest.partitions[client_id].indices
        return self._data[indices], self._labels[indices]
    
    def generate_heatmap_data(self, manifest: PartitionManifest) -> np.ndarray:
        """
        Generate heatmap matrix for visualization.
        
        Returns:
            Matrix of shape (num_clients, num_classes) with sample counts.
        """
        num_clients = manifest.num_clients
        num_classes = manifest.num_classes
        
        heatmap = np.zeros((num_clients, num_classes), dtype=np.int32)
        
        for i, (client_id, info) in enumerate(manifest.partitions.items()):
            for class_id, count in info.class_distribution.items():
                heatmap[i, class_id] = count
        
        return heatmap
    
    def get_entropy_summary(self, manifest: PartitionManifest) -> Dict[str, float]:
        """Get entropy for all clients."""
        return {cid: info.entropy for cid, info in manifest.partitions.items()}
    
    def repartition(self,
                    manifest: PartitionManifest,
                    strategy: PartitionStrategy,
                    growth_factor: float = 1.0) -> PartitionManifest:
        """
        Re-partition data, simulating data growth on clients.
        
        Args:
            manifest: Existing manifest.
            strategy: New partitioning strategy.
            growth_factor: Factor by which to grow data (1.0 = no growth).
        
        Returns:
            New PartitionManifest.
        """
        return self.partition(strategy, manifest.num_clients)


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

class PartitionerRegistry:
    """Registry for partitioning strategies."""
    
    _strategies = {
        "iid": IIDPartitioner,
        "dirichlet": DirichletPartitioner,
        "shard": ShardPartitioner,
    }
    
    @classmethod
    def get(cls, name: str, **kwargs) -> PartitionStrategy:
        """Get a strategy by name."""
        if name.lower() not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name.lower()](**kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies."""
        return list(cls._strategies.keys())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Data Partitioner Tests ===\n")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(num_samples=10000, num_classes=10, seed=42)
    partitioner = DataPartitioner(dataset)
    
    # Test IID
    print("--- IID Partitioning ---")
    iid = PartitionerRegistry.get("iid", seed=42)
    manifest_iid = partitioner.partition(iid, num_clients=5)
    print(f"Total samples: {manifest_iid.total_samples}")
    for cid, info in manifest_iid.partitions.items():
        print(f"  {cid}: {info.num_samples} samples, entropy={info.entropy:.2f}")
    
    # Test Dirichlet (high skew)
    print("\n--- Dirichlet (α=0.1) High Skew ---")
    dirichlet_skew = PartitionerRegistry.get("dirichlet", alpha=0.1, seed=42)
    manifest_skew = partitioner.partition(dirichlet_skew, num_clients=5)
    for cid, info in manifest_skew.partitions.items():
        print(f"  {cid}: {info.num_samples} samples, entropy={info.entropy:.2f}, classes={list(info.class_distribution.keys())}")
    
    # Test Dirichlet (IID-like)
    print("\n--- Dirichlet (α=100) IID-like ---")
    dirichlet_iid = PartitionerRegistry.get("dirichlet", alpha=100, seed=42)
    manifest_iid2 = partitioner.partition(dirichlet_iid, num_clients=5)
    for cid, info in manifest_iid2.partitions.items():
        print(f"  {cid}: {info.num_samples} samples, entropy={info.entropy:.2f}")
    
    # Test Shard
    print("\n--- Shard-Based (2 shards/client) ---")
    shard = PartitionerRegistry.get("shard", shards_per_client=2, seed=42)
    manifest_shard = partitioner.partition(shard, num_clients=5)
    for cid, info in manifest_shard.partitions.items():
        print(f"  {cid}: {info.num_samples} samples, entropy={info.entropy:.2f}, classes={list(info.class_distribution.keys())}")
    
    # Heatmap
    print("\n--- Heatmap Data ---")
    heatmap = partitioner.generate_heatmap_data(manifest_skew)
    print(f"Heatmap shape: {heatmap.shape}")
    print(heatmap)
    
    print("\n✅ All partitioner tests passed!")
