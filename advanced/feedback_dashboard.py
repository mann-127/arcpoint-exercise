"""Real-time dashboard for monitoring the routing system.

Displays:
- Live routing decisions
- Prediction accuracy over time
- Drift detection status
- Anomaly alerts
- A/B test results
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import deque

# Page config
st.set_page_config(
    page_title="Arcpoint Context Engine Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Arcpoint Context Engine Dashboard")
st.markdown("*Real-time monitoring for intelligent routing*")

# Sidebar
st.sidebar.header("⚙️ Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.5, 5.0, 1.0)
show_raw_data = st.sidebar.checkbox("Show Raw Data", False)

# Initialize session state
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = deque(maxlen=100)
    st.session_state.predictions = deque(maxlen=100)
    st.session_state.actuals = deque(maxlen=100)
    st.session_state.decisions = deque(maxlen=100)
    st.session_state.anomalies = []
    st.session_state.step = 0

# Simulate data generation
def generate_metrics(step):
    """Generate simulated metrics for demo."""
    # Simulate traffic pattern with occasional spikes
    is_spike = (step % 30 >= 20) and (step % 30 <= 25)
    
    load = 250 + np.random.normal(0, 20) if is_spike else 100 + np.random.normal(0, 10)
    base_latency = 50 + load * 0.8
    if is_spike:
        base_latency += (step % 30 - 20) * 30  # Degradation during spike
    actual_latency = base_latency + np.random.normal(0, 10)
    
    # Simple prediction (mock)
    predicted_latency = load * 0.7 + 60 + np.random.normal(0, 15)
    
    decision = "REROUTE" if predicted_latency > 300 else "PRIMARY"
    
    return {
        'timestamp': datetime.now(),
        'load': load,
        'actual_latency': actual_latency,
        'predicted_latency': predicted_latency,
        'decision': decision,
        'error': abs(predicted_latency - actual_latency),
        'is_anomaly': is_spike and load > 280
    }

# Main layout
col1, col2, col3, col4 = st.columns(4)

# Generate new data point
metrics = generate_metrics(st.session_state.step)
st.session_state.step += 1

# Store history
st.session_state.metrics_history.append(metrics)
st.session_state.predictions.append(metrics['predicted_latency'])
st.session_state.actuals.append(metrics['actual_latency'])
st.session_state.decisions.append(metrics['decision'])

if metrics['is_anomaly']:
    st.session_state.anomalies.append({
        'time': metrics['timestamp'],
        'load': metrics['load'],
        'latency': metrics['actual_latency']
    })

# KPI Cards
with col1:
    st.metric(
        "Current Load",
        f"{metrics['load']:.0f} req/s",
        f"{metrics['load'] - 100:.0f}" if len(st.session_state.metrics_history) > 1 else None
    )

with col2:
    st.metric(
        "Actual Latency",
        f"{metrics['actual_latency']:.0f} ms",
        f"{metrics['actual_latency'] - list(st.session_state.actuals)[-2] if len(st.session_state.actuals) > 1 else 0:.0f} ms"
    )

with col3:
    errors = [m['error'] for m in st.session_state.metrics_history]
    mae = np.mean(errors) if errors else 0
    st.metric("Prediction MAE", f"{mae:.1f} ms")

with col4:
    accuracy = sum(1 for m in st.session_state.metrics_history 
                   if (m['predicted_latency'] > 300) == (m['actual_latency'] > 300)) / max(len(st.session_state.metrics_history), 1)
    st.metric("Decision Accuracy", f"{accuracy*100:.1f}%")

# Status indicator
if metrics['decision'] == "REROUTE":
    st.warning(f"⚠️ **REROUTING** - Predicted latency: {metrics['predicted_latency']:.0f}ms")
else:
    st.success(f"✅ **PRIMARY** - Predicted latency: {metrics['predicted_latency']:.0f}ms")

# Charts
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Latency Over Time")
    
    if len(st.session_state.metrics_history) > 5:
        df = pd.DataFrame(list(st.session_state.metrics_history))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['actual_latency'],
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['predicted_latency'],
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))
        fig.add_hline(y=300, line_dash="dot", line_color="red", annotation_text="Threshold")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting data...")

with col_right:
    st.subheader("📊 Prediction Error Distribution")
    
    if len(st.session_state.metrics_history) > 10:
        errors = [m['error'] for m in st.session_state.metrics_history]
        fig = px.histogram(errors, nbins=20, title="")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting data...")

# Second row
st.markdown("---")
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("🔄 Routing Decisions")
    
    if st.session_state.decisions:
        decisions_df = pd.DataFrame({
            'decision': list(st.session_state.decisions)
        })
        counts = decisions_df['decision'].value_counts()
        
        fig = px.pie(
            values=counts.values, 
            names=counts.index,
            color=counts.index,
            color_discrete_map={'PRIMARY': 'green', 'REROUTE': 'orange'}
        )
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_right2:
    st.subheader("🚨 Anomaly Alerts")
    
    if st.session_state.anomalies:
        for anomaly in st.session_state.anomalies[-5:]:
            st.error(f"**{anomaly['time'].strftime('%H:%M:%S')}** - "
                    f"Load: {anomaly['load']:.0f}, Latency: {anomaly['latency']:.0f}ms")
    else:
        st.success("No anomalies detected")

# Drift Detection Status
st.markdown("---")
st.subheader("📉 Model Health")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    # Simulated drift metric
    drift_score = np.mean([m['error'] for m in list(st.session_state.metrics_history)[-20:]]) if len(st.session_state.metrics_history) > 20 else 0
    drift_status = "🟢 Stable" if drift_score < 50 else "🟡 Elevated" if drift_score < 100 else "🔴 Drift Detected"
    st.metric("Drift Status", drift_status)

with col_d2:
    st.metric("Samples Processed", len(st.session_state.metrics_history))

with col_d3:
    st.metric("Anomaly Rate", f"{len(st.session_state.anomalies) / max(st.session_state.step, 1) * 100:.1f}%")

# Raw data
if show_raw_data and st.session_state.metrics_history:
    st.markdown("---")
    st.subheader("📋 Raw Data")
    df = pd.DataFrame(list(st.session_state.metrics_history)[-20:])
    st.dataframe(df)

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
