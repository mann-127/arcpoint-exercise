# 🧠 Arcpoint Context Engine (ACE)
### *A Predictive Circuit Breaker for Intelligent Routing*

**Option Selected:** Option 3 (ML-Augmented Routing)

## 📋 Overview
ACE (Arcpoint Context Engine) is a prototype **Context Layer** designed to bring foresight to the routing engine.

In high-scale systems, reactive health checks are often too slow—by the time a heartbeat fails, thousands of requests may have already degraded. ACE moves from **Reactive** to **Proactive** by predicting system degradation *before* it impacts the user.

It answers the core question: **"What capacity risks are we taking in the next 5 minutes?"**

---

## 🏗️ Architecture

The system follows a **Stream-to-Inference** pattern designed for sub-millisecond overhead:

1. **Ingest (Mock):** Synthetic data generator simulates realistic traffic patterns with periodic load spikes and correlated latency degradation.
2. **State Management:** A sliding-window `FeatureStore` maintains recent cluster state in memory (production would use Redis).
3. **Inference Engine:** Calculates real-time derivatives (rate of change) and queries the pre-trained Random Forest model.
4. **Policy Layer:** Converts predictions into routing decisions based on configurable thresholds.

**Code Quality:** Industry-standard docstrings, structured logging, and modular design for production readiness.

---

## 🛠️ Design Decisions & Trade-offs

### 1. Why "Predictive" instead of "Reactive"?
Traditional health checks (e.g., a heartbeat every 30s) are insufficient for an intelligent control plane.
* **My Solution:** I trained a model to treat `Current Load` and `Latency Velocity` as leading indicators. This allows us to shed load *before* the backend collapses.

### 2. Model Choice: Random Forest vs. Deep Learning
I deliberately chose a **Random Forest Regressor** over LSTM/Transformers for this prototype.
* **Reasoning:** In the critical routing path, **inference latency** is the bottleneck. A decision tree inference is $\approx O(\text{depth})$, taking microseconds. A deep learning model would introduce unacceptable overhead (50ms+) for a routing decision that needs to be instant.
* **Trade-off:** We sacrifice long-term sequence memory for immediate speed and interpretability.

### 3. Handling Async Quality Scores
The prompt noted that "Quality scores are available async (hours later)."
* **Strategy:** I analyzed the data (see `notebooks/exploration.ipynb`) and found `latency` is a strong proxy for quality degradation. The model uses `latency` as a real-time proxy feature, while `quality_score` is used only for offline training/labeling.

### 4. Threshold Tuning (Product Sense)
I tuned the reroute threshold to **300ms**.
* **Reasoning:** While the critical failure point might be higher (500ms+), a high-reliability system should be biased towards false positives ("better safe than sorry"). It is better to unnecessarily reroute traffic than to let users suffer an outage.

### 5. The "Cold Start" Problem
The system defaults to simple Round-Robin routing if the Feature Store has fewer than 5 data points (e.g., after a restart), ensuring high availability even when the "Intelligence" layer is warming up.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.12+
* `make` (optional, for convenience)

### 1. Setup
Install dependencies.
```bash
make setup
# OR: pip install -r requirements.txt
```

### 2. Generate Synthetic Data
Create the "Universe" of logs to train the model.

```bash
make data
# OR: python data/mock_generator.py
```

### 3. Train the Brain
Train the Random Forest predictor.

```bash
make train
# OR: python src/model.py
```

**Expected Output:**
- MAE: ~60-65ms (model predicts within 60ms on average)
- R² Score: 0.50-0.55 (explains 50%+ of variance)

### 4. Run the Routing Engine
Simulate a stream of incoming requests and watch the router make decisions.

```bash
make run
# OR: python src/router.py
```

**Expected Output:** You'll see the circuit breaker in action:
- First 4-5 timesteps: Cold start (ROUND_ROBIN)
- When load spikes to 250: Proactive ⚠️ REROUTE decisions
- When load returns to 100: Back to ✅ PRIMARY

### 5. Explore the Analysis (Optional)
Open [notebooks/exploration.ipynb](notebooks/exploration.ipynb) to see:
- Visual proof of load-latency correlation (r=0.95)
- Leading indicator analysis showing slope spikes before latency peaks
- Statistical validation of feature engineering choices

---

## 📂 Project Structure

```plaintext
arcpoint-exercise/
├── data/
│   ├── mock_generator.py      # Synthetic data generator with causal patterns
│   └── historical_logs.csv    # Generated training data (gitignored)
├── src/
│   ├── features.py            # In-memory feature store with sliding window
│   ├── model.py               # Model training pipeline with logging
│   └── router.py              # Intelligent router with predictive circuit breaker
├── models/
│   └── latency_predictor.pkl  # Trained Random Forest model (gitignored)
├── notebooks/
│   └── exploration.ipynb      # EDA: load-latency correlation analysis
├── Makefile                   # Development shortcuts
├── requirements.txt           # Python dependencies with version pins
├── .gitignore                 # Excludes generated files and caches
└── README.md                  # Project documentation
```

---

## � Implementation Details

### Code Organization
- **Modular design:** Each component (feature store, model, router) is independently testable
- **Type hints & docstrings:** All public methods have comprehensive documentation
- **Logging:** Structured logging at INFO level for operational visibility
- **Constants:** Magic numbers extracted as class constants for maintainability

### Key Features
- **Cold start handling:** Graceful degradation when insufficient data is available
- **Time-series aware:** Training uses temporal split (not random) to prevent data leakage
- **Fast inference:** Random Forest provides microsecond-level predictions
- **Dependency management:** Pinned versions prevent compatibility issues

---

## �🔮 Future Improvements

1. **Feedback Loop:** Implement a reinforcement learning reward signal based on the eventual `quality_score` to auto-tune the threshold.

2. **Redis Implementation:** Replace the in-memory Pandas FeatureStore with Redis TimeSeries for production persistence.

3. **Shadow Mode:** Deploy the model in "shadow mode" to verify the `risk_score` calibration against live traffic before enabling active blocking.

4. **Multi-Backend Support:** Extend to consider multiple backends simultaneously and optimize routing across the fleet.

5. **A/B Testing Framework:** Build infrastructure to test different threshold values and model architectures in production.
