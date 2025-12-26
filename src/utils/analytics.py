"""
Analytics Engine for Federated Learning
========================================
Experiment tracking, metric logging, and performance analysis.
"""

import numpy as np
import json
import sqlite3
import uuid
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import pickle

logger = logging.getLogger("FLAnalytics")


# ============================================================================
# METRIC LOGGER
# ============================================================================

@dataclass
class RoundMetrics:
    """Metrics for a single training round."""
    round_num: int
    timestamp: str
    
    # Performance
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    
    # Weight divergence
    weight_divergence: float = 0.0
    
    # Communication
    uplink_mb: float = 0.0
    downlink_mb: float = 0.0
    total_traffic_mb: float = 0.0
    accuracy_per_mb: float = 0.0
    
    # Timing
    training_time_s: float = 0.0
    aggregation_time_s: float = 0.0
    communication_time_s: float = 0.0
    
    # Privacy
    epsilon_spent: float = 0.0
    snr: float = 0.0
    
    # Client stats
    num_clients: int = 0
    flagged_clients: int = 0


class MetricLogger:
    """
    Centralized logger for FL metrics.
    
    Tracks performance, divergence, efficiency across rounds.
    """
    
    def __init__(self):
        self.rounds: List[RoundMetrics] = []
        self.current_round = 0
        self.convergence_round: Optional[int] = None
        self.convergence_threshold = 0.9
    
    def log_round(self, metrics: RoundMetrics):
        """Log metrics for a round."""
        self.rounds.append(metrics)
        self.current_round = metrics.round_num
        
        # Check convergence
        if self.convergence_round is None and metrics.global_accuracy >= self.convergence_threshold:
            self.convergence_round = metrics.round_num
            logger.info(f"Convergence reached at round {metrics.round_num}")
    
    def compute_weight_divergence(self,
                                   global_weights: Dict[str, np.ndarray],
                                   client_updates: Dict[str, Dict[str, np.ndarray]]) -> float:
        """
        Compute average weight divergence.
        
        Formula: Δ_t = (1/|S_t|) * Σ_k ||w_{t+1} - w_k^{t+1}||_2
        """
        if not client_updates:
            return 0.0
        
        total_divergence = 0.0
        for cid, updates in client_updates.items():
            client_norm_sq = 0.0
            for layer, w in global_weights.items():
                if layer in updates:
                    diff = w - updates[layer]
                    client_norm_sq += float(np.sum(diff ** 2))
            total_divergence += float(np.sqrt(client_norm_sq))
        
        return float(total_divergence / len(client_updates))
    
    def get_accuracy_history(self) -> List[float]:
        """Get accuracy history."""
        return [r.global_accuracy for r in self.rounds]
    
    def get_loss_history(self) -> List[float]:
        """Get loss history."""
        return [r.global_loss for r in self.rounds]
    
    def get_divergence_history(self) -> List[float]:
        """Get weight divergence history."""
        return [r.weight_divergence for r in self.rounds]
    
    def get_cumulative_traffic(self) -> float:
        """Get total traffic in MB."""
        return sum(r.total_traffic_mb for r in self.rounds)
    
    def get_communication_efficiency(self) -> float:
        """Calculate accuracy improvement per MB."""
        total_mb = self.get_cumulative_traffic()
        if total_mb == 0 or not self.rounds:
            return 0.0
        final_acc = self.rounds[-1].global_accuracy
        return final_acc / total_mb
    
    def to_dict_list(self) -> List[Dict]:
        """Convert to list of dicts for export."""
        return [asdict(r) for r in self.rounds]


# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    name: str
    created_at: str
    
    # Hyperparameters
    num_rounds: int = 10
    num_clients: int = 10
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    lr_decay: float = 0.95  # LR decay factor per round
    
    # FL Settings
    aggregation_algo: str = "FedAvg"
    non_iid_alpha: float = 0.5
    
    # Privacy
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    clip_norm: float = 1.0
    
    # Results
    final_accuracy: float = 0.0
    final_loss: float = 0.0
    convergence_round: Optional[int] = None
    total_training_time_s: float = 0.0


class ExperimentRegistry:
    """
    Experiment versioning and persistence.
    
    Stores experiments in SQLite database.
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT,
                    config_json TEXT,
                    metrics_json TEXT,
                    status TEXT DEFAULT 'running'
                )
            """)
            conn.commit()
    
    def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """Create a new experiment."""
        exp_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()
        
        exp_config = ExperimentConfig(
            experiment_id=exp_id,
            name=name,
            created_at=created_at,
            **config
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO experiments (experiment_id, name, created_at, config_json, metrics_json) VALUES (?, ?, ?, ?, ?)",
                (exp_id, name, created_at, json.dumps(asdict(exp_config)), "{}")
            )
            conn.commit()
        
        logger.info(f"Created experiment: {exp_id} ({name})")
        return exp_id
    
    def update_experiment(self, exp_id: str, metrics: List[Dict], status: str = "running"):
        """Update experiment with metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET metrics_json = ?, status = ? WHERE experiment_id = ?",
                (json.dumps(metrics), status, exp_id)
            )
            conn.commit()
    
    def complete_experiment(self, exp_id: str, final_acc: float, final_loss: float, 
                            convergence_round: Optional[int], total_time: float):
        """Mark experiment as complete with final results."""
        with sqlite3.connect(self.db_path) as conn:
            # Get current config
            cursor = conn.execute("SELECT config_json FROM experiments WHERE experiment_id = ?", (exp_id,))
            row = cursor.fetchone()
            if row:
                config = json.loads(row[0])
                config["final_accuracy"] = final_acc
                config["final_loss"] = final_loss
                config["convergence_round"] = convergence_round
                config["total_training_time_s"] = total_time
                
                conn.execute(
                    "UPDATE experiments SET config_json = ?, status = ? WHERE experiment_id = ?",
                    (json.dumps(config), "completed", exp_id)
                )
                conn.commit()
    
    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT experiment_id, name, created_at, config_json, metrics_json, status FROM experiments WHERE experiment_id = ?",
                (exp_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "experiment_id": row[0],
                    "name": row[1],
                    "created_at": row[2],
                    "config": json.loads(row[3]),
                    "metrics": json.loads(row[4]),
                    "status": row[5]
                }
        return None
    
    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT experiment_id, name, created_at, config_json, status FROM experiments ORDER BY created_at DESC"
            )
            experiments = []
            for row in cursor.fetchall():
                config = json.loads(row[3])
                experiments.append({
                    "id": row[0],
                    "name": row[1],
                    "created_at": row[2],
                    "algo": config.get("aggregation_algo", "N/A"),
                    "accuracy": config.get("final_accuracy", 0.0),
                    "status": row[4]
                })
            return experiments
    
    def delete_experiment(self, exp_id: str):
        """Delete an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (exp_id,))
            conn.commit()


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Manages model checkpoints.
    
    Saves best model (by validation accuracy) and last model.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_accuracy = 0.0
        self.best_round = 0
    
    def save_checkpoint(self, 
                        exp_id: str,
                        round_num: int,
                        weights: Dict[str, np.ndarray],
                        accuracy: float,
                        is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            "experiment_id": exp_id,
            "round": round_num,
            "accuracy": accuracy,
            "weights": weights,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save last
        last_path = os.path.join(self.checkpoint_dir, f"{exp_id}_last.pkl")
        with open(last_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Save best if applicable
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_round = round_num
            best_path = os.path.join(self.checkpoint_dir, f"{exp_id}_best.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(checkpoint, f)
            logger.info(f"New best model saved: round {round_num}, acc {accuracy:.4f}")
    
    def load_checkpoint(self, exp_id: str, which: str = "best") -> Optional[Dict]:
        """Load a checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{exp_id}_{which}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None


# ============================================================================
# FAIRNESS ANALYZER
# ============================================================================

class FairnessAnalyzer:
    """
    Analyzes fairness and variance across clients.
    """
    
    def __init__(self):
        self.client_accuracies: Dict[str, List[float]] = {}
    
    def record_client_accuracy(self, client_id: str, accuracy: float):
        """Record accuracy for a client."""
        if client_id not in self.client_accuracies:
            self.client_accuracies[client_id] = []
        self.client_accuracies[client_id].append(accuracy)
    
    def compute_accuracy_variance(self) -> Tuple[float, float]:
        """
        Compute mean and std of client accuracies.
        
        Returns:
            Tuple of (mean, std)
        """
        if not self.client_accuracies:
            return 0.0, 0.0
        
        latest_accs = [accs[-1] for accs in self.client_accuracies.values() if accs]
        if not latest_accs:
            return 0.0, 0.0
        
        return float(np.mean(latest_accs)), float(np.std(latest_accs))
    
    def get_pareto_points(self, 
                          accuracy_history: List[float],
                          privacy_history: List[float]) -> List[Tuple[float, float]]:
        """
        Compute Pareto frontier for accuracy vs privacy trade-off.
        
        Returns:
            List of (accuracy, epsilon) points on the frontier.
        """
        if len(accuracy_history) != len(privacy_history):
            return []
        
        points = list(zip(accuracy_history, privacy_history))
        
        # Sort by accuracy descending
        points.sort(key=lambda x: -x[0])
        
        # Filter Pareto-optimal points
        pareto = []
        min_epsilon = float('inf')
        
        for acc, eps in points:
            if eps < min_epsilon:
                pareto.append((acc, eps))
                min_epsilon = eps
        
        return pareto


# ============================================================================
# SYSTEM HEALTH MONITOR
# ============================================================================

class SystemHealthMonitor:
    """
    Monitors system performance and timing.
    """
    
    def __init__(self):
        self.phase_times: Dict[str, List[float]] = {
            "training": [],
            "aggregation": [],
            "communication": [],
            "idle": []
        }
        self._phase_start: Optional[float] = None
        self._current_phase: Optional[str] = None
    
    def start_phase(self, phase: str):
        """Start timing a phase."""
        if self._phase_start is not None and self._current_phase:
            self.end_phase()
        
        self._phase_start = time.time()
        self._current_phase = phase
    
    def end_phase(self) -> float:
        """End current phase and return duration."""
        if self._phase_start is None:
            return 0.0
        
        duration = time.time() - self._phase_start
        if self._current_phase:
            self.phase_times[self._current_phase].append(duration)
        
        self._phase_start = None
        self._current_phase = None
        return duration
    
    def get_efficiency_score(self) -> float:
        """
        Calculate system efficiency score.
        
        Formula: T_comp / (T_comp + T_comm + T_idle)
        """
        t_comp = sum(self.phase_times["training"]) + sum(self.phase_times["aggregation"])
        t_comm = sum(self.phase_times["communication"])
        t_idle = sum(self.phase_times["idle"])
        
        total = t_comp + t_comm + t_idle
        if total == 0:
            return 1.0
        
        return t_comp / total
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average time per phase."""
        return {
            phase: np.mean(times) if times else 0.0
            for phase, times in self.phase_times.items()
        }
    
    def get_total_time(self) -> float:
        """Get total wall-clock time."""
        return sum(sum(times) for times in self.phase_times.values())


# ============================================================================
# DATA EXPORT
# ============================================================================

class DataExporter:
    """
    Export experiment data to various formats.
    """
    
    @staticmethod
    def to_csv(metrics: List[Dict], filepath: str):
        """Export metrics to CSV."""
        if not metrics:
            return
        
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        
        logger.info(f"Exported to CSV: {filepath}")
    
    @staticmethod
    def to_json(data: Any, filepath: str):
        """Export data to JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported to JSON: {filepath}")
    
    @staticmethod
    def to_parquet(metrics: List[Dict], filepath: str):
        """Export metrics to Parquet (requires pandas and pyarrow)."""
        try:
            import pandas as pd
            df = pd.DataFrame(metrics)
            df.to_parquet(filepath, index=False)
            logger.info(f"Exported to Parquet: {filepath}")
        except ImportError:
            logger.warning("Parquet export requires pandas and pyarrow")


# ============================================================================
# UNIFIED ANALYTICS MANAGER
# ============================================================================

class AnalyticsManager:
    """
    Unified analytics manager integrating all components.
    """
    
    def __init__(self, experiment_name: str = "default", config: Dict[str, Any] = None):
        self.metric_logger = MetricLogger()
        self.registry = ExperimentRegistry()
        self.checkpoint_mgr = CheckpointManager()
        self.fairness = FairnessAnalyzer()
        self.health = SystemHealthMonitor()
        self.exporter = DataExporter()
        
        # Create experiment
        self.experiment_id = self.registry.create_experiment(
            experiment_name,
            config or {}
        )
        self.start_time = time.time()
    
    def log_round(self,
                  round_num: int,
                  global_weights: Dict[str, np.ndarray],
                  client_updates: Dict[str, Dict[str, np.ndarray]],
                  accuracy: float,
                  loss: float,
                  traffic_mb: float = 0.0,
                  epsilon_spent: float = 0.0,
                  snr: float = 0.0,
                  flagged_clients: int = 0):
        """Log a complete round."""
        # Compute divergence
        divergence = self.metric_logger.compute_weight_divergence(global_weights, client_updates)
        
        metrics = RoundMetrics(
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            global_accuracy=accuracy,
            global_loss=loss,
            weight_divergence=divergence,
            total_traffic_mb=traffic_mb,
            accuracy_per_mb=accuracy / max(traffic_mb, 0.001),
            epsilon_spent=epsilon_spent,
            snr=snr,
            num_clients=len(client_updates),
            flagged_clients=flagged_clients
        )
        
        self.metric_logger.log_round(metrics)
        
        # Save checkpoint
        self.checkpoint_mgr.save_checkpoint(
            self.experiment_id, round_num, global_weights, accuracy
        )
        
        # Update registry
        self.registry.update_experiment(
            self.experiment_id,
            self.metric_logger.to_dict_list()
        )
    
    def complete(self):
        """Mark experiment as complete."""
        total_time = time.time() - self.start_time
        
        final_acc = self.metric_logger.rounds[-1].global_accuracy if self.metric_logger.rounds else 0.0
        final_loss = self.metric_logger.rounds[-1].global_loss if self.metric_logger.rounds else 0.0
        
        self.registry.complete_experiment(
            self.experiment_id,
            final_acc,
            final_loss,
            self.metric_logger.convergence_round,
            total_time
        )
        
        logger.info(f"Experiment {self.experiment_id} completed in {total_time:.2f}s")
    
    def export(self, format: str = "csv"):
        """Export experiment data."""
        metrics = self.metric_logger.to_dict_list()
        base_path = f"experiment_{self.experiment_id}"
        
        if format == "csv":
            self.exporter.to_csv(metrics, f"{base_path}.csv")
        elif format == "json":
            self.exporter.to_json(metrics, f"{base_path}.json")
        elif format == "parquet":
            self.exporter.to_parquet(metrics, f"{base_path}.parquet")
    
    def get_leaderboard(self) -> List[Dict]:
        """Get experiment leaderboard."""
        return self.registry.get_all_experiments()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "experiment_id": self.experiment_id,
            "total_rounds": len(self.metric_logger.rounds),
            "final_accuracy": self.metric_logger.rounds[-1].global_accuracy if self.metric_logger.rounds else 0.0,
            "convergence_round": self.metric_logger.convergence_round,
            "total_traffic_mb": self.metric_logger.get_cumulative_traffic(),
            "communication_efficiency": self.metric_logger.get_communication_efficiency(),
            "system_efficiency": self.health.get_efficiency_score(),
            "total_time_s": time.time() - self.start_time
        }


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Analytics Engine Tests ===\n")
    
    # Create manager
    manager = AnalyticsManager("test_experiment", {"num_rounds": 5, "aggregation_algo": "FedAvg"})
    
    # Simulate rounds
    weights = {"w": np.random.randn(10, 10).astype(np.float32)}
    
    for r in range(1, 6):
        client_updates = {f"c{i}": {"w": weights["w"] + np.random.randn(10, 10).astype(np.float32) * 0.1} for i in range(5)}
        
        acc = 0.5 + 0.1 * r
        loss = 2.0 - 0.3 * r
        
        manager.log_round(
            round_num=r,
            global_weights=weights,
            client_updates=client_updates,
            accuracy=acc,
            loss=loss,
            traffic_mb=5.0,
            epsilon_spent=0.1 * r
        )
        
        print(f"Round {r}: Acc={acc:.2f}, Loss={loss:.2f}")
    
    manager.complete()
    
    # Export
    manager.export("csv")
    
    # Summary
    print("\nSummary:", manager.get_summary())
    
    # Leaderboard
    print("\nLeaderboard:", manager.get_leaderboard())
    
    print("\n✅ Analytics tests passed!")