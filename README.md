# 🛰️ FedVisualizer

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=flat-square&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**High-Fidelity Federated Learning Research Platform**

Real-time visualization • Differential Privacy • Secure Aggregation • Byzantine Robustness

---

## 🎯 Overview

FedVisualizer is a **production-ready research platform** for Federated Learning experimentation. Built with a cyberpunk-inspired UI, it provides researchers with powerful tools to visualize, analyze, and optimize FL training in real-time.

### Key Capabilities

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
STREAMLIT DASHBOARD (app.py)
├── KPI Cards (10 metrics)
├── Network Topology (Plotly)
├── Convergence Charts
└── Console Logs

src/core/
├── server.py      → Federated Aggregation
├── client.py      → Client Simulator
└── security.py    → DP, SecAgg, Byzantine Defense

src/utils/
├── data_partitioner.py  → IID/Dirichlet/Shard
├── analytics.py         → Metrics & Export
└── network.py           → Serialization
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

Navigate to `http://localhost:8501`

---

## 📊 Dashboard Features

### KPI Cards

**Row 1:** Round, Clients, Accuracy, Traffic (MB)

**Row 2:** Learning Rate (η), Weight Divergence (Δw), Speed (%/r), Privacy (ε)

**Row 3:** Work/Wait Ratio, Throughput (kS/s)

### Bottleneck Analysis

| Status | Idle % | Meaning |
|--------|--------|---------|
| 🔴 UI Overhead | >80% | Streamlit refresh is bottleneck |
| 🟡 Communication Bound | 50-80% | Increase local epochs |
| 🟢 Balanced | 20-50% | Healthy compute-to-wait ratio |
| ✅ Compute Heavy | <20% | Optimal for research |

### Success Alert

When target accuracy is reached, displays:
- Final Accuracy
- Privacy Budget (ε) spent
- Total Traffic (MB)
- Rounds Completed

---

## 🔬 Mathematical Foundations

### Federated Averaging (FedAvg)

```
w(t+1) = Σ (nk/n) * wk(t+1)
```

### Server Momentum

```
v(t+1) = β * v(t) + (1-β) * Δ(t+1)
w(t+1) = w(t) + η * v(t+1)
```

### Cosine Similarity Weight Divergence

```
D_cos(wt, wk) = 1 - (wt · wk) / (||wt|| ||wk||)
```

### Differential Privacy (Gaussian Mechanism)

```
w_noisy = w + N(0, σ²I)
σ = C * sqrt(2 * ln(1.25/δ)) / ε
```

### Non-IID Data (Dirichlet Distribution)

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
| GaussianMechanism | Adds calibrated noise to gradients |
| L2GradientClipper | Bounds gradient sensitivity |
| RDPAccountant | Tracks privacy budget across rounds |

### Secure Aggregation

Zero-sum masking protocol where server sees masked updates but aggregate cancels masks.

### Byzantine Robustness

| Defense | Strategy |
|---------|----------|
| Krum | Selects update closest to k neighbors |
| Median | Coordinate-wise median aggregation |
| Anomaly Detection | Cosine similarity outlier detection |

---

## ⚙️ Configuration

### Parameters

| Category | Parameter | Range | Default |
|----------|-----------|-------|---------|
| Parameters | Rounds | 1-100 | 20 |
| Parameters | Local Epochs (E) | 1-20 | 5 |
| Parameters | Algorithm | FedAvg/FedProx/FedAdam | FedAvg |
| Optimization | Learning Rate (η) | 0.001-0.1 | 0.01 |
| Optimization | LR Decay | 0.8-1.0 | 0.95 |
| Optimization | Batch Size (B) | 32/64/128 | 64 |
| Optimization | Target Accuracy | 0.8-0.99 | 0.92 |
| Optimization | Server Momentum (β) | 0.0-0.99 | 0.9 |
| Privacy | Differential Privacy | on/off | on |
| Privacy | Target ε | 0.1-20.0 | 5.0 |
| Privacy | Noise Multiplier (σ) | 0.5-10.0 | 2.5 |
| Privacy | Clip Norm (C) | 0.5-5.0 | 1.0 |
| Network | Clients | 3-50 | 10 |
| Network | Non-IID α | 0.01-10.0 | 0.5 |

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
```

### AnalyticsManager

```python
from src.utils.analytics import AnalyticsManager

analytics = AnalyticsManager("experiment_1", {"algo": "FedAvg"})
analytics.log_round(round_num=1, accuracy=0.9, loss=0.5)
analytics.complete()
analytics.export("csv")
```

---

## 📁 Project Structure

```
Fed-Visualizer/
├── app.py                 # Main Streamlit Dashboard
├── requirements.txt       # Dependencies
├── LICENSE                # MIT License
├── README.md              # Documentation
├── src/
│   ├── core/
│   │   ├── server.py      # Federated Aggregation Server
│   │   ├── client.py      # Client Simulator
│   │   ├── security.py    # DP, SecAgg, Byzantine Defense
│   │   └── fl_algorithms.py
│   └── utils/
│       ├── data_partitioner.py
│       ├── analytics.py
│       └── network.py
└── checkpoints/           # Model Checkpoints
```

---

## 📚 References

- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
- **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
- **Differential Privacy**: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
- **Secure Aggregation**: Bonawitz et al., "Practical Secure Aggregation for Federated Learning" (CCS 2017)

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

**Built with ❤️ for the Federated Learning Research Community**
