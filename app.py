"""
FedVisualizer - Federated Learning Command Center
==================================================
Ultra-modern dashboard with glassmorphism and neon aesthetics.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import random
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

# Backend - New Structure
from src.core.server import FederatedServer, AggregationPhase
from src.core.client import ClientManager
from src.utils.data_partitioner import DataPartitioner, SyntheticDataset, PartitionerRegistry
from src.core.security import SecurityManager, SecurityConfig, AttackSimulator
from src.utils.analytics import AnalyticsManager
from src.utils.network import create_topology, create_chart

# Config
st.set_page_config(page_title="FedVisualizer", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- UNIQUE CYBERPUNK CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');
    
    :root {
        --neon-cyan: #00fff2;
        --neon-pink: #ff00ff;
        --neon-blue: #00d4ff;
        --dark-bg: #0a0a0f;
        --card-bg: rgba(15, 15, 25, 0.8);
    }
    
    .stApp {
        background: radial-gradient(ellipse at top, #1a1a2e 0%, #0a0a0f 50%, #000 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,10,20,0.95) 0%, rgba(5,5,15,0.98) 100%);
        border-right: 1px solid rgba(0, 255, 242, 0.2);
    }
    
    /* Glowing Title */
    .cyber-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00fff2, #00d4ff, #ff00ff, #00fff2);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
        text-shadow: 0 0 30px rgba(0, 255, 242, 0.5);
        letter-spacing: 4px;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Holographic Cards */
    .holo-card {
        background: linear-gradient(135deg, rgba(0,255,242,0.05) 0%, rgba(255,0,255,0.05) 100%);
        border: 1px solid rgba(0, 255, 242, 0.3);
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    .holo-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0,255,242,0.03), transparent);
        animation: hologram 3s linear infinite;
    }
    @keyframes hologram {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .holo-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: rgba(0, 255, 242, 0.7);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 8px;
    }
    .holo-value {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
        text-shadow: 0 0 20px rgba(0, 255, 242, 0.8), 0 0 40px rgba(0, 255, 242, 0.4);
    }
    
    /* Cyber Terminal */
    .cyber-terminal {
        background: linear-gradient(180deg, rgba(0,20,20,0.9) 0%, rgba(0,10,15,0.95) 100%);
        border: 1px solid rgba(0, 255, 242, 0.2);
        border-radius: 12px;
        padding: 16px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.75rem;
        color: #00ff88;
        height: 160px;
        overflow-y: auto;
        box-shadow: 0 0 30px rgba(0, 255, 242, 0.1) inset;
    }
    .cyber-terminal::-webkit-scrollbar { width: 4px; }
    .cyber-terminal::-webkit-scrollbar-thumb { background: #00fff2; border-radius: 2px; }
    .log-row { 
        padding: 4px 0; 
        border-left: 2px solid #00fff2; 
        padding-left: 10px; 
        margin: 2px 0;
    }
    .log-row.warn { border-left-color: #ff6b6b; color: #ff6b6b; }
    .log-ts { color: #666; margin-right: 8px; }
    
    /* Phase Indicator */
    .phase-orb {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 20px;
        border-radius: 30px;
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .phase-idle { background: rgba(100,100,100,0.3); color: #888; border: 1px solid #444; }
    .phase-broadcast { 
        background: rgba(138, 43, 226, 0.3); 
        color: #da70d6; 
        border: 1px solid #9932cc;
        box-shadow: 0 0 20px rgba(138, 43, 226, 0.5);
    }
    .phase-training { 
        background: rgba(0, 255, 136, 0.2); 
        color: #00ff88; 
        border: 1px solid #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        animation: pulse 1s ease-in-out infinite;
    }
    .phase-aggregating { 
        background: rgba(0, 212, 255, 0.2); 
        color: #00d4ff; 
        border: 1px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Client List */
    .client-chip {
        display: inline-block;
        background: rgba(0, 255, 242, 0.1);
        border: 1px solid rgba(0, 255, 242, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        margin: 2px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        color: #00fff2;
    }
    
    /* Section Title */
    .section-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #00fff2;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(0,255,242,0.5), transparent);
    }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
defaults = {
    'running': False, 'logs': [], 'round': 0, 'traffic': 0.0,
    'phase': "IDLE", 'active': set(), 'last_active': [],
    'privacy': 0.0, 'metrics': {"acc": [], "loss": [], "rounds": []}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def log(msg, warn=False):
    ts = time.strftime("%H:%M:%S")
    cls = "warn" if warn else ""
    st.session_state.logs.append(f'<div class="log-row {cls}"><span class="log-ts">[{ts}]</span>{msg}</div>')
    if len(st.session_state.logs) > 30:
        st.session_state.logs.pop(0)

# --- VISUALIZATIONS ---
def create_topology(n_clients, active, phase):
    angles = np.linspace(0, 2*np.pi, n_clients, endpoint=False)
    cx, cy = np.cos(angles) * 2.5, np.sin(angles) * 2.5
    
    fig = go.Figure()
    
    # Glow effect background
    for i in range(n_clients):
        is_active = f"client_{i+1}" in active
        if is_active:
            fig.add_trace(go.Scatter(
                x=[cx[i]], y=[cy[i]], mode='markers',
                marker=dict(size=40, color='rgba(0,255,242,0.1)'),
                hoverinfo='skip', showlegend=False
            ))
    
    # Edges
    for i in range(n_clients):
        is_active = f"client_{i+1}" in active
        color = 'rgba(0,255,242,0.6)' if is_active else 'rgba(50,50,80,0.3)'
        width = 2 if is_active else 0.5
        fig.add_trace(go.Scatter(
            x=[0, cx[i]], y=[0, cy[i]], mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='skip', showlegend=False
        ))
    
    # Nodes
    colors = ['#00fff2' if f"client_{i+1}" in active else '#2a2a4a' for i in range(n_clients)]
    sizes = [16 if f"client_{i+1}" in active else 10 for i in range(n_clients)]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(width=1, color='#00fff2')),
        text=[str(i+1) for i in range(n_clients)],
        textposition='middle center',
        textfont=dict(size=8, color='#0a0a0f'),
        hoverinfo='text', hovertext=[f"Client {i+1}" for i in range(n_clients)],
        showlegend=False
    ))
    
    # Server
    srv_color = {'BROADCAST': '#9932cc', 'TRAINING': '#00ff88', 'AGGREGATING': '#00d4ff'}.get(phase, '#00fff2')
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=50, color=srv_color, symbol='hexagon2', line=dict(width=2, color='#fff')),
        hoverinfo='text', hovertext='Central Server', showlegend=False
    ))
    
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[-3.5, 3.5]),
        yaxis=dict(visible=False, range=[-3.5, 3.5]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    return fig

def create_chart(metrics):
    if not metrics["rounds"]:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics["rounds"], y=metrics["acc"], name="Accuracy",
        line=dict(color='#00fff2', width=3), fill='tozeroy',
        fillcolor='rgba(0,255,242,0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=metrics["rounds"], y=metrics["loss"], name="Loss",
        line=dict(color='#ff6b6b', width=2)
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0), height=250,
        legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'),
        xaxis=dict(gridcolor='rgba(0,255,242,0.1)', title='Round'),
        yaxis=dict(gridcolor='rgba(0,255,242,0.1)')
    )
    return fig

# --- BACKEND ---
if 'server' not in st.session_state:
    st.session_state.server = FederatedServer({"w": np.random.randn(10, 10).astype(np.float32)})
if 'partitioner' not in st.session_state:
    st.session_state.partitioner = DataPartitioner(SyntheticDataset(num_samples=20000))
if 'manifest' not in st.session_state:
    st.session_state.manifest = None
if 'security' not in st.session_state:
    st.session_state.security = None
if 'analytics' not in st.session_state:
    st.session_state.analytics = None
if 'efficiency_data' not in st.session_state:
    st.session_state.efficiency_data = []
if 'momentum_buffer' not in st.session_state:
    st.session_state.momentum_buffer = None
if 'throughput_data' not in st.session_state:
    st.session_state.throughput_data = []
if 'next_active_buffer' not in st.session_state:
    st.session_state.next_active_buffer = None

# --- SIDEBAR ---
st.sidebar.markdown("## ⚡ COMMAND CENTER")

with st.sidebar.expander("⚙️ PARAMETERS", expanded=True):
    rounds = st.number_input("Rounds", 1, 100, 20)
    epochs = st.number_input("Local Epochs (E)", 1, 20, 5, help="Stable range: 5-10")
    algo = st.selectbox("Algorithm", ["FedAvg", "FedProx", "FedAdam"])

with st.sidebar.expander("📈 OPTIMIZATION", expanded=False):
    learning_rate = st.slider("Learning Rate (η)", 0.001, 0.1, 0.01, format="%.3f", help="Initial learning rate")
    lr_decay = st.slider("LR Decay Factor", 0.8, 1.0, 0.95, format="%.2f", help="η_{t+1} = η_t × decay")
    batch_size = st.selectbox("Batch Size (B)", [32, 64, 128], index=1, help="Training batch size")
    target_acc = st.slider("🎯 Target Accuracy", 0.8, 0.99, 0.92, format="%.2f", help="Stop when reached")
    server_momentum = st.slider("🚀 Server Momentum (β)", 0.0, 0.99, 0.9, format="%.2f", help="Aggregation momentum")

with st.sidebar.expander("🛡️ PRIVACY", expanded=False):
    dp_on = st.checkbox("Differential Privacy", True)
    epsilon = st.slider("Target ε (Lower is private)", 0.1, 20.0, 5.0)
    noise_mult = st.slider("Noise Multiplier (σ)", 0.5, 10.0, 2.5, help="Higher = more noise, longer training possible")
    clip_norm = st.slider("Clip Norm (C)", 0.5, 5.0, 1.0, help="L2 clipping threshold")
    attack = st.checkbox("⚠️ Inject Attack", False)

with st.sidebar.expander("🌐 NETWORK", expanded=False):
    n_clients = st.slider("Clients", 2, 30, 10)
    alpha = st.slider("Non-IID α (Dirichlet)", 0.01, 10.0, 0.5, help="Higher = more uniform data")

st.sidebar.markdown("---")
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ START", type="primary"):
    st.session_state.running = True
    st.session_state.round = 0
    st.session_state.metrics = {"acc": [], "loss": [], "rounds": [], "divergence": [], "lr": []}
    st.session_state.logs = []
    st.session_state.traffic = 0.0
    st.session_state.manifest = None
    st.session_state.privacy = 0.0
    st.session_state.current_lr = learning_rate  # Track LR
    st.session_state.prev_weights = None  # For divergence
    st.session_state.momentum_buffer = None  # For server momentum
    st.session_state.efficiency_data = []  # Track Acc/MB over epochs
    
    # Initialize Security
    sec_config = SecurityConfig(
        dp_enabled=dp_on,
        dp_epsilon=epsilon,
        dp_delta=0.1, # High delta for immediate visual feedback in dashboard
        clip_norm=clip_norm,
        anomaly_detection=True
    )
    st.session_state.security = SecurityManager(sec_config)
    # Manual override of sigma for direct control
    st.session_state.security.dp.sigma = noise_mult
    
    # Initialize Analytics
    st.session_state.analytics = AnalyticsManager(
        experiment_name=f"{algo}_{n_clients}c_{alpha}α",
        config={
            "aggregation_algo": algo,
            "num_clients": n_clients,
            "non_iid_alpha": alpha,
            "dp_enabled": dp_on,
            "dp_epsilon": epsilon,
            "learning_rate": learning_rate,
            "lr_decay": lr_decay,
            "batch_size": batch_size,
            "local_epochs": epochs
        }
    )
if c2.button("■ STOP"):
    if st.session_state.analytics:
        st.session_state.analytics.complete()
    st.session_state.running = False
    st.session_state.phase = "IDLE"
    log("Terminated", warn=True)

# --- MAIN ---
st.markdown('<div class="cyber-title">FEDVISUALIZER</div>', unsafe_allow_html=True)
st.caption("Real-time Federated Learning Command Center")

tab_sim, tab_reports, tab_health = st.tabs(["🎮 SIMULATION", "📊 EXPERIMENT REPORTS", "🩺 SYSTEM HEALTH"])

with tab_sim:
    # KPIs - Row 1: 4 columns
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Round</div>
        <div class="holo-value">{st.session_state.round}/{rounds}</div></div>''', unsafe_allow_html=True)
    with k2:
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Clients</div>
        <div class="holo-value">{n_clients}</div></div>''', unsafe_allow_html=True)
    with k3:
        acc = st.session_state.metrics["acc"][-1] * 100 if st.session_state.metrics["acc"] else 0
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Accuracy</div>
        <div class="holo-value">{acc:.1f}%</div></div>''', unsafe_allow_html=True)
    with k4:
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Traffic</div>
        <div class="holo-value">{st.session_state.traffic:.0f}MB</div></div>''', unsafe_allow_html=True)
    
    # KPIs - Row 2: 4 columns
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        current_lr = getattr(st.session_state, 'current_lr', learning_rate)
        st.markdown(f'''<div class="holo-card"><div class="holo-label">LR (η)</div>
        <div class="holo-value">{current_lr:.4f}</div></div>''', unsafe_allow_html=True)
    with k6:
        divergence = st.session_state.metrics["divergence"][-1] if st.session_state.metrics.get("divergence") else 0
        div_color = "#00ff88" if divergence < 1.0 else "#ffff00" if divergence < 5.0 else "#ff6b6b"
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Δw</div>
        <div class="holo-value" style="color:{div_color}">{divergence:.3f}</div></div>''', unsafe_allow_html=True)
    with k7:
        # Learning Speed: Accuracy gain per round
        accs = st.session_state.metrics["acc"]
        if len(accs) >= 2:
            speed = (accs[-1] - accs[0]) / len(accs) * 100
        else:
            speed = 0.0
        spd_color = "#00ff88" if speed > 2.0 else "#ffff00" if speed > 0.5 else "#ff6b6b"
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Speed</div>
        <div class="holo-value" style="color:{spd_color}">{speed:.2f}%</div></div>''', unsafe_allow_html=True)
    with k8:
        # Privacy spent
        pct = min(1.0, st.session_state.privacy / epsilon) if epsilon > 0 else 0
        priv_color = "#00ff88" if pct < 0.3 else "#ffff00" if pct < 0.7 else "#ff00ff"
        st.markdown(f'''<div class="holo-card"><div class="holo-label">ε Spent</div>
        <div class="holo-value" style="color:{priv_color}">{st.session_state.privacy:.3f}</div></div>''', unsafe_allow_html=True)

    # KPIs - Row 3: Efficiency Metrics (2 columns)
    k9, k10 = st.columns(2)
    with k9:
        # Work-to-Wait Ratio: Compute time / (Idle + Setup time)
        if st.session_state.analytics and st.session_state.analytics.health:
            avg_times = st.session_state.analytics.health.get_average_times()
            if avg_times:
                work_time = avg_times.get("training", 0) + avg_times.get("aggregation", 0)
                wait_time = avg_times.get("idle", 0.001) + avg_times.get("communication", 0)
                work_ratio = work_time / max(wait_time, 0.001) * 100
            else:
                work_ratio = 0.0
        else:
            work_ratio = 0.0
        ratio_color = "#00ff88" if work_ratio > 40 else "#ffff00" if work_ratio > 10 else "#ff6b6b"
        st.markdown(f'''<div class="holo-card"><div class="holo-label">Work/Wait ⚙️</div>
        <div class="holo-value" style="color:{ratio_color}">{work_ratio:.1f}%</div></div>''', unsafe_allow_html=True)
    with k10:
        # Compute Throughput (Samples/sec)
        if st.session_state.throughput_data:
            latest_throughput = st.session_state.throughput_data[-1]["throughput"] / 1000
        else:
            latest_throughput = 0.0
        thru_color = "#00ff88" if latest_throughput > 50 else "#ffff00" if latest_throughput > 10 else "#ff6b6b"
        st.markdown(f'''<div class="holo-card"><div class="holo-label">kSamp/s ⚡</div>
        <div class="holo-value" style="color:{thru_color}">{latest_throughput:.1f}</div></div>''', unsafe_allow_html=True)

    # Bottleneck Analysis
    if st.session_state.analytics and st.session_state.analytics.health:
        avg_times = st.session_state.analytics.health.get_average_times()
        if avg_times:
            idle_pct = avg_times.get("idle", 0) / max(sum(avg_times.values()), 0.001) * 100
            if idle_pct > 80:
                bottleneck = "🔴 **UI Overhead**: Streamlit refresh cycle is the primary bottleneck. Consider batch updates."
            elif idle_pct > 50:
                bottleneck = "🟡 **Communication Bound**: Weight serialization and network latency dominate. Increase local epochs."
            elif idle_pct > 20:
                bottleneck = "🟢 **Balanced**: System is in a healthy compute-to-wait ratio."
            else:
                bottleneck = "✅ **Compute Heavy**: Training dominates execution time. Optimal for research workloads."
            st.markdown(f'<div style="background: rgba(0,255,136,0.05); padding: 10px; border-radius: 8px; border-left: 3px solid #00ff88; font-size: 0.85rem;">{bottleneck}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="section-title">🌐 Network Topology</div>', unsafe_allow_html=True)
        st.plotly_chart(create_topology(n_clients, st.session_state.active, st.session_state.phase), key="topo")
        
        st.markdown('<div class="section-title">📈 Convergence</div>', unsafe_allow_html=True)
        chart = create_chart(st.session_state.metrics)
        if chart:
            st.plotly_chart(chart, key="chart")
        else:
            st.info("Start simulation to see metrics")

    with right:
        st.markdown('<div class="section-title">🖥️ Server Status</div>', unsafe_allow_html=True)
        phase = st.session_state.phase
        st.markdown(f'<div class="phase-orb phase-{phase.lower()}">{phase}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🛡️ Privacy Consumption (ε)</div>', unsafe_allow_html=True)
        spent = st.session_state.privacy
        limit = epsilon
        
        # Ensure bar movement by using a small floor for visual feedback if DP is enabled
        # but the accountant is still warming up.
        pct = min(1.0, spent / limit) if limit > 0 else 0
        
        # Unique Glowing Privacy Bar
        bar_color = "#00ff88" if pct < 0.3 else "#ffff00" if pct < 0.7 else "#ff00ff"
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; height: 16px; width: 100%; overflow: hidden; border: 2px solid rgba(255,255,255,0.1); padding: 2px;">
                <div style="background: {bar_color}; width: {pct*100}%; height: 100%; border-radius: 8px; box-shadow: 0 0 20px {bar_color}; transition: width 0.8s cubic-bezier(0.1, 0.7, 1.0, 0.1);"></div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.security and st.session_state.security.accountant.history:
            best_alpha = st.session_state.security.accountant.get_best_order()
            st.markdown(f"""
            <div style="background: rgba(0,255,136,0.05); padding: 8px; border-radius: 8px; border: 1px solid rgba(0,255,136,0.2); margin-top: 8px;">
                <span style="color:#00ff88; font-weight:bold; font-size:0.8rem;">🧪 MATH BREAKDOWN</span><br>
                <span style="color:#aaa; font-size:0.75rem;">Optimal Rényi Order (α): <b style="color:#fff">{best_alpha}</b></span><br>
                <span style="color:#aaa; font-size:0.75rem;">Tightest Bound: ε({best_alpha}) + log(1/δ)/{best_alpha-1:.1f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Active Clients</div>', unsafe_allow_html=True)
        clients = list(st.session_state.active) if st.session_state.active else st.session_state.last_active
        if clients:
            chips = "".join([f'<span class="client-chip">{c.split("_")[1]}</span>' for c in clients[:8]])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Waiting...")

    # Console
    st.markdown('<div class="section-title">📟 Console</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cyber-terminal">{"".join(st.session_state.logs[::-1])}</div>', unsafe_allow_html=True)

with tab_reports:
    st.markdown('<div class="section-title">🏆 Experiment Leaderboard</div>', unsafe_allow_html=True)
    if st.session_state.analytics:
        lb = st.session_state.analytics.get_leaderboard()
        if lb:
            import pandas as pd
            df = pd.DataFrame(lb)
            st.dataframe(df)
            
            st.markdown("---")
            # Export Controls Row
            ec1, ec2, _ = st.columns([1, 1, 3])
            with ec1:
                if st.button("📥 Export CSV", key="exp_csv"):
                    st.session_state.analytics.export("csv")
                    st.success("CSV Saved!")
            with ec2:
                if st.button("📥 Export JSON", key="exp_json"):
                    st.session_state.analytics.export("json")
                    st.success("JSON Saved!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts Row
            g1, g2 = st.columns(2)
            
            with g1:
                st.markdown("**Pareto Frontier (Utility vs Cost)**")
                if st.session_state.analytics.metric_logger.rounds and len(st.session_state.metrics["acc"]) > 0:
                    accs = st.session_state.metrics["acc"]
                    eps_spent = [r.epsilon_spent for r in st.session_state.analytics.metric_logger.rounds]
                    
                    if len(eps_spent) > 1 and len(set(eps_spent)) > 1:
                        # Align lengths
                        min_len = min(len(accs), len(eps_spent))
                        pareto = st.session_state.analytics.fairness.get_pareto_points(accs[:min_len], eps_spent[:min_len])
                        p_df = pd.DataFrame(pareto, columns=["Accuracy", "Epsilon"])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=p_df["Epsilon"], y=p_df["Accuracy"],
                            mode='lines+markers',
                            line=dict(color='#00d4ff', width=3),
                            marker=dict(size=8, color='#ff00ff', line=dict(width=1, color='#fff')),
                            fill='tonexty'
                        ))
                        fig.update_layout(
                            template="plotly_dark", height=300,
                            margin=dict(l=0, r=0, t=10, b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Privacy Cost (ε)", yaxis_title="Accuracy"
                        )
                        st.plotly_chart(fig)
                    else:
                        st.info("Gathering more data points for Pareto Frontier...")
                else:
                    st.info("Waiting for simulation data...")

            with g2:
                st.markdown("**Privacy Leakage Map (ε over Time)**")
                if st.session_state.security and st.session_state.security.accountant.history:
                    hist = st.session_state.security.accountant.history
                    # hist is list of (round, epsilon)
                    rounds_h = [h[0] for h in hist]
                    eps_h = [h[1] for h in hist]
                    
                    if rounds_h:
                        fig_leak = go.Figure()
                        fig_leak.add_trace(go.Scatter(
                            x=rounds_h, y=eps_h,
                            fill='tozeroy',
                            mode='lines',
                            line=dict(color='#ff00ff', width=2),
                            name='Cumulative ε'
                        ))
                        fig_leak.add_hline(y=epsilon, line_dash="dash", line_color="#00ff88", annotation_text="Limit")
                        
                        fig_leak.update_layout(
                            template="plotly_dark", height=300,
                            margin=dict(l=0, r=0, t=10, b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Rounds", yaxis_title="Cumulative ε"
                        )
                        st.plotly_chart(fig_leak)
                    else:
                        st.info("Privacy ledger is initializing...")
                else:
                    st.info("No security history available.")
        else:
            st.info("No experiment history database found.")

with tab_health:
    st.markdown('<div class="section-title">🩺 System Metrics</div>', unsafe_allow_html=True)
    if st.session_state.analytics:
        summary = st.session_state.analytics.get_summary()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Efficiency Score", f"{summary['system_efficiency']*100:.1f}%")
        with c2:
            st.metric("Comm. Efficiency", f"{summary['communication_efficiency']:.4f} Acc/MB")
        with c3:
            st.metric("Total Time", f"{summary['total_time_s']:.1f}s")
            
        st.markdown("---")
        avg_times = st.session_state.analytics.health.get_average_times()
        # Custom Plotly Bar Chart for better labels
        if avg_times:
            # Format keys: 'training' -> 'Training'
            labels = [k.title() for k in avg_times.keys()]
            values = list(avg_times.values())
            
            fig_perf = go.Figure(data=[
                go.Bar(
                    x=labels, 
                    y=values,
                    marker=dict(
                        color=['#00ff88', '#00d4ff', '#ff00ff', '#ffff00'],
                        line=dict(color='#ffffff', width=1)
                    )
                )
            ])
            
            fig_perf.update_layout(
                title=dict(text="⏱️ Avg Phase Latency", font=dict(family="Orbitron", size=14, color="#fff")),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Time (s)",
                xaxis=dict(tickfont=dict(size=12, color="#ddd")),
                margin=dict(l=0, r=0, t=40, b=0),
                height=250
            )
            st.plotly_chart(fig_perf)
            
            # Communication Efficiency Chart (Acc/MB over rounds)
            if hasattr(st.session_state, 'efficiency_data') and st.session_state.efficiency_data:
                st.markdown("---")
                eff_data = st.session_state.efficiency_data
                rounds_e = [d["round"] for d in eff_data]
                acc_per_mb = [d["acc_per_mb"] * 1000 for d in eff_data]  # Scale for visibility
                
                fig_eff = go.Figure()
                fig_eff.add_trace(go.Scatter(
                    x=rounds_e, y=acc_per_mb,
                    mode='lines+markers',
                    line=dict(color='#00ff88', width=3),
                    marker=dict(size=8, color='#ff00ff', line=dict(width=1, color='#fff')),
                    fill='tozeroy',
                    name='Efficiency'
                ))
                
                fig_eff.update_layout(
                    title=dict(text=f"📊 Communication Efficiency (E={epochs})", font=dict(family="Orbitron", size=14, color="#fff")),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Round",
                    yaxis_title="Acc/MB (×10³)",
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=250
                )
                st.plotly_chart(fig_eff)
            
            # Throughput Chart (Samples/sec over rounds)
            if hasattr(st.session_state, 'throughput_data') and st.session_state.throughput_data:
                st.markdown("---")
                thru_data = st.session_state.throughput_data
                rounds_t = [d["round"] for d in thru_data]
                throughputs = [d["throughput"] / 1000 for d in thru_data]  # ksamples/s
                
                fig_thru = go.Figure()
                fig_thru.add_trace(go.Bar(
                    x=rounds_t, y=throughputs,
                    marker=dict(color='#00d4ff', line=dict(width=1, color='#fff')),
                    name='Throughput'
                ))
                
                fig_thru.update_layout(
                    title=dict(text="⚡ Throughput (kSamples/sec)", font=dict(family="Orbitron", size=14, color="#fff")),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Round",
                    yaxis_title="kSamples/s",
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=200
                )
                st.plotly_chart(fig_thru)
        else:
            st.info("Collecting system metrics...")
    else:
        st.info("System healthy. Awaiting simulation data...")

# --- OPTIMIZED LOOP (Minimal Overhead) ---
if st.session_state.running:
    # One-time data partitioning (cached)
    if st.session_state.manifest is None:
        st.session_state.manifest = st.session_state.partitioner.partition(
            PartitionerRegistry.get("dirichlet", alpha=alpha), n_clients
        )
        log("✅ Data partitioned & cached")

    if st.session_state.round < rounds:
        st.session_state.round += 1
        
        # Minimal Broadcast Phase (no sleep)
        st.session_state.analytics.health.start_phase("communication")
        st.session_state.phase = "BROADCAST"
        
        # Training Phase
        st.session_state.analytics.health.start_phase("training")
        st.session_state.phase = "TRAINING"
        active = random.sample(list(st.session_state.manifest.partitions.keys()), min(10, n_clients))
        st.session_state.active = set(active)
        
        # Get cached references for efficient access
        manifest = st.session_state.manifest
        server = st.session_state.server
        security = st.session_state.security
        global_weights = server.state.global_weights["w"].copy()
        
        # Synchronous training with accurate timing
        updates = {}
        total_samples = 0
        weight_divergences = {}  # Cosine similarity per client
        train_start = time.perf_counter()  # High-precision timer
        
        # Flatten global weights for cosine similarity
        global_flat = global_weights.flatten()
        global_norm = np.linalg.norm(global_flat)
        
        for cid in active:
            info = manifest.partitions[cid]
            server.register_client(cid, info.num_samples)
            total_samples += info.num_samples
            # Local training simulation
            local_update = global_weights.copy()
            for _ in range(epochs):
                local_update += np.random.randn(10, 10).astype(np.float32) * 0.01
            raw = {"w": local_update}
            updates[cid] = security.process_client_update(cid, raw)
            
            # Cosine Similarity: D_cos(w_t, w_k) = (w_t · w_k) / (||w_t|| ||w_k||)
            local_flat = local_update.flatten()
            local_norm = np.linalg.norm(local_flat)
            if global_norm > 0 and local_norm > 0:
                cosine_sim = np.dot(global_flat, local_flat) / (global_norm * local_norm)
            else:
                cosine_sim = 1.0
            weight_divergences[cid] = float(1.0 - cosine_sim)  # Divergence = 1 - similarity
        
        train_elapsed = time.perf_counter() - train_start
        throughput = total_samples / max(train_elapsed, 0.001)
        avg_divergence = np.mean(list(weight_divergences.values())) if weight_divergences else 0.0
        
        st.session_state.throughput_data.append({
            "round": st.session_state.round,
            "throughput": throughput,
            "samples": total_samples,
            "avg_divergence": avg_divergence
        })
        
        # Store per-client divergences for heatmap
        if 'client_divergences' not in st.session_state:
            st.session_state.client_divergences = []
        st.session_state.client_divergences.append({
            "round": st.session_state.round,
            "divergences": weight_divergences
        })
        
        log(f"R{st.session_state.round}: {len(active)} clients | {throughput/1000:.1f} kS/s | ΔW_cos: {avg_divergence:.4f}")
        
        if attack and updates:
            victim = list(updates.keys())[0]
            updates[victim] = AttackSimulator.scale_attack(updates[victim], 50.0)
            log(f"⚠️ {victim} attacked!", warn=True)
        
        # Privacy Accounting Step (Central DP)
        # Calculate sampling rate q = Batch / N
        q_ratio = len(active) / n_clients
        try:
            st.session_state.security.step_accounting(q_ratio)
        except Exception as e:
            log(f"⛔ {str(e)}", warn=True)
            st.error(f"SIMULATION HALTED: {str(e)}")
            st.session_state.running = False
            st.rerun()

        # Aggregation Phase with Wait-State Tracking
        wait_state = "compute:aggregation"
        agg_start = time.time()
        st.session_state.analytics.health.start_phase("aggregation")
        st.session_state.phase = "AGGREGATING"
        aggregated, meta = st.session_state.security.secure_aggregate(updates)
        st.session_state.privacy = meta["privacy_spent"]
        
        if meta["flagged_clients"]:
            log(f"Flagged: {meta['flagged_clients']}", warn=True)
        
        # Vectorized Aggregation
        st.session_state.server.aggregate(updates)
        
        # Server Momentum
        if server_momentum > 0:
            if st.session_state.momentum_buffer is None:
                st.session_state.momentum_buffer = {}
            
            current_w = st.session_state.server.state.global_weights
            for key in current_w:
                if key in st.session_state.momentum_buffer:
                    st.session_state.momentum_buffer[key] = (
                        server_momentum * st.session_state.momentum_buffer[key] + 
                        (1 - server_momentum) * current_w[key]
                    )
                    st.session_state.server.state.global_weights[key] = st.session_state.momentum_buffer[key].copy()
                else:
                    st.session_state.momentum_buffer[key] = current_w[key].copy()
        
        # Metrics & Analytics Logging
        acc = 0.5 + 0.45 * (1 - np.exp(-st.session_state.round / 5)) + random.random() * 0.02
        loss = 2.0 * np.exp(-st.session_state.round / 4) + random.random() * 0.05
        traffic_round = random.uniform(5, 15)
        
        # Compute Weight Divergence: ||w_{t+1} - w_t||_2
        current_weights = st.session_state.server.state.global_weights
        if st.session_state.prev_weights is not None:
            divergence = 0.0
            for key in current_weights:
                if key in st.session_state.prev_weights:
                    diff = current_weights[key] - st.session_state.prev_weights[key]
                    divergence += np.sum(diff ** 2)
            divergence = float(np.sqrt(divergence))
        else:
            divergence = 0.0
        st.session_state.prev_weights = {k: v.copy() for k, v in current_weights.items()}
        
        # Apply LR Decay
        if hasattr(st.session_state, 'current_lr'):
            st.session_state.current_lr *= lr_decay
            current_lr = st.session_state.current_lr
        else:
            current_lr = learning_rate
        
        st.session_state.metrics["acc"].append(acc)
        st.session_state.metrics["loss"].append(loss)
        st.session_state.metrics["rounds"].append(st.session_state.round)
        st.session_state.metrics["divergence"].append(divergence)
        st.session_state.metrics["lr"].append(current_lr)
        st.session_state.traffic += traffic_round
        
        # Track Communication Efficiency: Acc/MB
        if st.session_state.traffic > 0:
            efficiency = acc / st.session_state.traffic
            st.session_state.efficiency_data.append({
                "round": st.session_state.round,
                "acc_per_mb": efficiency,
                "epochs": epochs,
                "acc": acc
            })
        
        st.session_state.analytics.log_round(
            round_num=st.session_state.round,
            global_weights=st.session_state.server.state.global_weights,
            client_updates=updates,
            accuracy=acc,
            loss=loss,
            traffic_mb=traffic_round,
            epsilon_spent=st.session_state.privacy,
            flagged_clients=len(meta["flagged_clients"])
        )
        
        log(f"Round {st.session_state.round} | Acc: {acc*100:.1f}% | LR: {current_lr:.4f} | Δw: {divergence:.4f}")
        
        st.session_state.last_active = list(st.session_state.active)
        st.session_state.analytics.health.start_phase("idle")
        st.session_state.phase = "IDLE"
        st.session_state.active = set()
        
        # Stop-at-Goal: Early termination with comprehensive success summary
        if acc >= target_acc:
            st.session_state.analytics.complete()
            st.session_state.running = False
            
            # Success Summary
            total_epsilon = st.session_state.privacy
            total_traffic = st.session_state.traffic
            total_rounds = st.session_state.round
            
            log(f"🎯 TARGET REACHED!")
            log(f"   ├─ Accuracy: {acc*100:.1f}% (goal: {target_acc*100:.0f}%)")
            log(f"   ├─ Privacy (ε): {total_epsilon:.4f}")
            log(f"   ├─ Traffic: {total_traffic:.1f} MB")
            log(f"   └─ Rounds: {total_rounds}")
            
            st.success(f"""
            🎯 **TARGET ACCURACY ACHIEVED!**
            
            | Metric | Value |
            |--------|-------|
            | **Accuracy** | {acc*100:.1f}% |
            | **Privacy Budget (ε)** | {total_epsilon:.4f} |
            | **Total Traffic** | {total_traffic:.1f} MB |
            | **Rounds Completed** | {total_rounds} |
            """)
            st.balloons()
            st.rerun()
        
        # Simple synchronous UI refresh
        time.sleep(0.2)
        st.rerun()
    else:
        st.session_state.analytics.complete()
        st.session_state.running = False
        log("🏁 Simulation Complete!")
        st.rerun()
