<p align="center">
  <img src="https://img.shields.io/badge/🛰️_FedVisualizer-v2.0-00ff88?style=for-the-badge&labelColor=1a1a2e" alt="FedVisualizer">
</p>

<h1 align="center">🛰️ FedVisualizer</h1>

<p align="center">
  <strong>High-Fidelity Federated Learning Research Platform</strong>
</p>

<p align="center">
  <em>Real-time visualization • Differential Privacy • Secure Aggregation • Byzantine Robustness</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/NumPy-1.21+-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Plotly-5.15+-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 🎯 Overview

FedVisualizer is a **production-ready research platform** for Federated Learning experimentation. Built with a cyberpunk-inspired UI, it provides researchers with powerful tools to visualize, analyze, and optimize FL training in real-time.

### ✨ Key Capabilities

| Feature | Description |
|---------|-------------|
| 🌐 **Real-time Topology** | Dynamic network graph showing server-client connections |
| 📊 **Live Metrics** | 10+ KPIs including accuracy, loss, throughput, and weight divergence |
| 🛡️ **Differential Privacy** | Rigorous RDP accounting with Gaussian mechanism |
| 🔐 **Secure Aggregation** | Zero-sum masking protocol for client privacy |
| ⚔️ **Byzantine Defense** | Krum and Median aggregation against adversaries |
| 📈 **Weight Divergence** | Cosine similarity tracking between global and local models |
| 🎯 **Bottleneck Analysis** | Intelligent system health diagnostics |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           🖥️ STREAMLIT DASHBOARD                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │  KPI Cards │  │  Topology  │  │  Charts    │  │  Console   │             │
│  │  (10 KPIs) │  │  (Plotly)  │  │  (Live)    │  │  (Logs)    │             │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘             │
├──────────────────────────────────────────────────────────────────────────────┤
│                           📦 src/core/                                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                              │
│  │  server.py │  │  client.py │  │ security.py│                              │
│  │ FedServer  │  │ClientManager│ │SecurityMgr │                              │
│  │ Aggregation│  │ Hardware   │  │ DP/SecAgg  │                              │
│  └────────────┘  └────────────┘  └────────────┘                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                           📦 src/utils/                                       │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐      │
│  │ data_partitioner.py│  │   analytics.py     │  │    network.py      │      │
│  │ IID/Dirichlet/Shard│  │ Metrics & Export   │  │ Serialization      │      │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yusufcalisir/Fed-Visualizer.git
cd Fed-Visualizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` 🚀

---

## 📊 Dashboard Features

### KPI Cards (3 Rows)

| Row | Metrics |
|-----|---------|
| **Row 1** | Round, Clients, Accuracy, Traffic (MB) |
| **Row 2** | Learning Rate (η), Weight Divergence (Δw), Speed (%/r), Privacy (ε) |
| **Row 3** | Work/Wait Ratio (⚙️), Throughput (kS/s ⚡) |

### Bottleneck Analysis

Real-time system health diagnostics:

| Status | Idle % | Meaning |
|--------|--------|---------|
| 🔴 **UI Overhead** | >80% | Streamlit refresh is bottleneck |
| 🟡 **Communication Bound** | 50-80% | Increase local epochs |
| 🟢 **Balanced** | 20-50% | Healthy compute-to-wait ratio |
| ✅ **Compute Heavy** | <20% | Optimal for research workloads |

### Success Alert

When target accuracy is reached:

```
🎯 TARGET ACCURACY ACHIEVED!

| Metric              | Value    |
|---------------------|----------|
| Accuracy            | 93.6%    |
| Privacy Budget (ε)  | 0.4521   |
| Total Traffic       | 125.3 MB |
| Rounds Completed    | 15       |
```

---

## 🔬 Mathematical Foundations

### Federated Averaging (FedAvg)

$$w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{t+1}$$

### Server Momentum

$$v_{t+1} = \beta v_t + (1-\beta) \Delta_{t+1}$$
$$w_{t+1} = w_t + \eta v_{t+1}$$

### Cosine Similarity Weight Divergence

$$D_{cos}(w_t, w_k) = 1 - \frac{w_t \cdot w_k}{\|w_t\| \|w_k\|}$$

### Differential Privacy (Gaussian Mechanism)

$$w_{noisy} = w + \mathcal{N}(0, \sigma^2 I)$$

$$\sigma = \frac{C \cdot \sqrt{2 \ln(1.25/\delta)}}{\epsilon}$$

### Non-IID Data (Dirichlet Distribution)

$$p_k \sim \text{Dir}(\alpha \cdot \mathbf{1})$$

| α Value | Data Distribution |
|---------|-------------------|
| α → 0 | Extreme heterogeneity (1 class/client) |
| α = 0.5 | Moderate heterogeneity (recommended) |
| α → ∞ | IID (uniform distribution) |

---

## 🛡️ Security Framework

### Differential Privacy

| Component | Purpose |
|-----------|---------|
| `GaussianMechanism` | Adds calibrated noise to gradients |
| `L2GradientClipper` | Bounds gradient sensitivity |
| `RDPAccountant` | Tracks privacy budget across rounds |

### Secure Aggregation

Zero-sum masking protocol:

$$\sum_{k=1}^{K} m_k = 0$$

Server sees masked updates: $w_k + m_k$

Aggregate cancels masks: $\sum_k (w_k + m_k) = \sum_k w_k$

### Byzantine Robustness

| Defense | Strategy |
|---------|----------|
| **Krum** | Selects update closest to k neighbors |
| **Median** | Coordinate-wise median aggregation |
| **Anomaly Detection** | Cosine similarity outlier detection |

---

## ⚙️ Configuration

### Sidebar Controls

| Category | Parameter | Range | Default |
|----------|-----------|-------|---------|
| **Parameters** | Rounds | 1-100 | 20 |
| | Local Epochs (E) | 1-20 | 5 |
| | Algorithm | FedAvg/FedProx/FedAdam | FedAvg |
| **Optimization** | Learning Rate (η) | 0.001-0.1 | 0.01 |
| | LR Decay | 0.8-1.0 | 0.95 |
| | Batch Size (B) | 32/64/128 | 64 |
| | Target Accuracy | 0.8-0.99 | 0.92 |
| | Server Momentum (β) | 0.0-0.99 | 0.9 |
| **Privacy** | Differential Privacy | on/off | on |
| | Target ε | 0.1-20.0 | 5.0 |
| | Noise Multiplier (σ) | 0.5-10.0 | 2.5 |
| | Clip Norm (C) | 0.5-5.0 | 1.0 |
| **Network** | Clients | 3-50 | 10 |
| | Non-IID α | 0.01-10.0 | 0.5 |

---

## 📈 Analytics & Reporting

### Experiment Reports Tab

- **Pareto Frontier**: Accuracy vs Privacy trade-off visualization
- **Privacy Leakage Map**: Cumulative ε growth over rounds
- **Export Options**: JSON, CSV, Parquet

### System Health Tab

- **Avg Phase Latency**: Bar chart with phase breakdown
- **Communication Efficiency**: Acc/MB over rounds
- **Throughput Chart**: kSamples/sec per round

---

## 🔌 API Reference

### FederatedServer

```python
from src.core.server import FederatedServer

server = FederatedServer(
    initial_weights={"w": np.random.randn(10, 10)},
    aggregation_strategy="fedavg"
)
server.register_client("client_1", num_samples=1000)
new_weights = server.aggregate(client_updates)
```

### SecurityManager

```python
from src.core.security import SecurityManager, SecurityConfig

config = SecurityConfig(
    dp_enabled=True,
    dp_epsilon=1.0,
    dp_delta=1e-5,
    clip_norm=1.0
)
security = SecurityManager(config)
processed = security.process_client_update("client_1", weights)
aggregated, meta = security.secure_aggregate(updates)
```

### AnalyticsManager

```python
from src.utils.analytics import AnalyticsManager

analytics = AnalyticsManager("experiment_1", {"algo": "FedAvg"})
analytics.log_round(round_num=1, accuracy=0.9, loss=0.5, ...)
analytics.complete()
analytics.export("csv")
```

---

## 📁 Project Structure

```
Fed-Visualizer/
├── app.py                    # 🖥️ Main Streamlit Dashboard
├── requirements.txt          # 📦 Dependencies
├── .gitignore               # 🚫 Git ignore rules
├── README.md                # 📖 Documentation
│
├── src/
│   ├── core/
│   │   ├── server.py        # 🖧 Federated Aggregation Server
│   │   ├── client.py        # 👤 Client Simulator
│   │   ├── security.py      # 🛡️ DP, SecAgg, Byzantine Defense
│   │   └── fl_algorithms.py # 📐 FedAvg, FedProx, FedAdam
│   │
│   └── utils/
│       ├── data_partitioner.py  # 📊 IID/Dirichlet/Shard
│       ├── analytics.py         # 📈 Metrics & Tracking
│       └── network.py           # 🌐 Serialization & Topology
│
└── checkpoints/             # 💾 Model Checkpoints (gitignored)
```

---

## 📚 References

| Paper | Citation |
|-------|----------|
| **FedAvg** | McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017) |
| **FedProx** | Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020) |
| **Differential Privacy** | Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014) |
| **Secure Aggregation** | Bonawitz et al., "Practical Secure Aggregation for Federated Learning" (CCS 2017) |
| **Byzantine FL** | Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ for the Federated Learning Research Community</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/⚡_Powered_by-Streamlit-FF4B4B?style=for-the-badge" alt="Streamlit">
  <img src="https://img.shields.io/badge/🔬_Research-Ready-00ff88?style=for-the-badge" alt="Research Ready">
</p>
#   F e d - V i s u a l i z e r  
 