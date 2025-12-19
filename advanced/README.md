# Advanced Features: Production-Grade System

Feedback loops, drift detection, anomaly detection, chaos engineering.

## Components

### Feedback Loop (`feedback_loop.py`)
Real-time outcome capture + online model updates + drift detection.

```python
from advanced.feedback_loop import FeedbackCollector, OnlineLearner, DriftDetector

collector = FeedbackCollector()
collector.record(predicted=100, actual=110)
learner = OnlineLearner()
learner.partial_fit(features, outcomes)
```

### Anomaly Detector (`anomaly_detector.py`)
Isolation Forest for detecting novel failure patterns.

### Chaos Simulator (`chaos_simulator.py`)
Inject failures to test system resilience.

```python
sim = ChaosSimulator()
result = sim.inject_failure('latency_spike', duration_seconds=5)
```

### Feedback Router (`feedback_router.py`)
Unified router combining all advanced features.

### Dashboard (`feedback_dashboard.py`)
Real-time observability with Streamlit.

```bash
streamlit run advanced/feedback_dashboard.py
```

## Architecture

```
ML Router + Feedback Loop + Anomaly Detection + Chaos Testing
         ↓
    Real-time Dashboard
```

## Production Use

1. Run `feedback_router.py` in production
2. Monitor dashboard for drift/anomalies
3. Chaos simulator for weekly resilience tests
4. Online learning continuously improves model

See main README.md for full context.
