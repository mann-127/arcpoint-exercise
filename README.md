# Arcpoint Context Engine

> Predictive Circuit Breaker for Intelligent Request Routing

## Problem Statement

Routing engines answer questions reactively:
- "Why did quality drop?" (after it happened)
- "Which backend should handle this?" (reactive load balancing)

Our mission: Answer proactively in real-time:
- "What's the current state of our model fleet?"
- "Which backend should I use for this request?"
- "Solution: Three Approaches

### Option 3: ML-Augmented Routing (Primary)
Predictive circuit breaker using Random Forest to detect latency degradation 5 minutes ahead.

**Key metrics:**
- MAE: 62ms (predictions within ±62ms)
- R²: 0.53 (explains 53% of variance)
- Inference latency: <1ms
- Decision: If predicted_latency > 300ms → REROUTE

### Option 2: LLM-Based Agent (Bonus)
Context API + Claude agent for explainable routing decisions with full reasoning trace.

**Trade-off:**
- Latency: ~500ms (slower but interpretable)
## Architecture

```
Request → Feature Store → ML Model → Routing Decision
            (Sliding         (Random    (Primary/
             Window)         Forest)    Secondary)
              ↓               ↓              ↓
         Feedback Loop ← Observe Outcome ← Execute
              ↓
        Drift Detection
              ↓
        Online Learner
```

## Design Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| **Random Forest over LSTM** | <1ms inference vs 50ms | Sacrificed sequence modeling for speed |
| **Load + Slope as features** | Slope is leading indicator (r=0.88) | Limited to 5-min prediction window |
| **300ms threshold** | Conservative: better safe than sorry | Some unnecessary reroutes during noise |
| **Online learning** | Adapts to production drift in real-time | Incremental updates vs full retraining |
| **Time-series split** | Respects temporal order, prevents data leakage | More realistic validation |

## Implementation

### Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate data & train
python3 data/mock_generator.py
python3 src/model.py

# Run router
python3 src/router.py

# Run dashboard
streamlit run advanced/feedback_dashboard.py

# Explore
jupyter notebook notebooks/exploration.ipynb
```
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
│ # Project Structure

```
arcpoint-exercise/
├── data/                        # Data generation
│   └── mock_generator.py        
├── src/                         # Core routing system
│   ├── features.py              
│   ├── model.py                 
│   └── router.py                
├── option2-agent/               # LLM-based agent approach
│   ├── context_api.py
│   ├── agent.py
│   └── prompts.py
├── advanced/                    # Production-grade features
│   ├── feedback_loop.py         
│   ├── anomaly_detector.py      
│   ├── chaos_simulator.py       
│   ├── feedback_router.py       
│   └── feedback_dashboard.py    
├── notebooks/                   # Data exploration
│   └── exploration.ipynb        
└── models/                      # Trained artifacts (gitignored)
### Key Features
- **Cold start handling:** Graceful degradation when insufficient data is available
- **Time-series aware:** Training uses temporal split (not random) to prevent data leakage
- **Fast inference:** Random Forest provides microsecond-level predictions
- **Dependency management:** Pinned versions prevent compatibility issues

---

## �🔮 Future Improvements

1. **Feedback Loop:** Implement a reinforcement learning reward signal based on the eventual `quality_score` to auto-tune the threshold.

2. **Redis Implementation:** Replace the in-memory Pandas FeatureStore with Redis TimeSeries for production persistence.

3. ~~**Shadow Mode:** Deploy the model in "shadow mode" to verify the `risk_score` calibration against live traffic before enabling active blocking.~~ ✅ **Implemented** — See `advanced/feedback_loop.py` with A/B testing.

4. **Multi-Backend Support:** Extend to consider multiple backends simultaneously and optimize routing across the fleet.

5. ~~**A/B Testing Framework:** Build infrastructure to test different threshold values and model architectures in production.~~ ✅ **Implemented** — See `advanced/feedback_loop.py` with t-test significance.

---

## 🔄 Advanced: Closed-Loop Feedback System

Beyond the basic ML predictor, I implemented a **production-grade feedback system** in [`advanced/`](advanced/).

### Components

| File | Purpose | Key Algorithm |
|------|---------|---------------|
| `feedback_loop.py` | Continuous learning | SGDRegressor (online), Page-Hinkley (drift) |
| `anomaly_detector.py` | Unusual pattern detection | Isolation Forest |
| `chaos_simulator.py` | Resilience testing | Netflix-style fault injection |
| `feedback_router.py` | Unified routing engine | Combines all components |
| `feedback_dashboard.py` | Real-time observability | Streamlit dashboard |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK ROUTER                               │
├─────────────────────────────────────────────────────────────────┤
│   ML Model ──▶ Decision Engine ──▶ Backend Router               │
│       ▲                                    │                     │
│       │            ┌──────────────┐        ▼                     │
│   Online      ◀────│   Feedback   │◀────  Actual                 │
│   Learner          │   Collector  │       Latency                │
│       │            └──────────────┘                              │
│       ▼                   │                                      │
│   Drift Detector    Anomaly Detector    Chaos Simulator          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Dashboard     │
                    │   (Streamlit)    │
                    └──────────────────┘
```

### Why This Matters

| Feature | Interview Signal |
|---------|------------------|
| Feedback Loop | "I understand models drift in production" |
| Online Learning | "I know batch training isn't enough" |
| Drift Detection | "I apply statistical rigor (Page-Hinkley)" |
| Anomaly Detection | "I handle edge cases proactively" |
| Chaos Engineering | "I think about failure modes" |
| Dashboard | "I build observable systems" |

### Quick Start

```bash
# Run the dashboard
streamlit run advanced/feedback_dashboard.py

# Run chaos test
python -c "from advanced.feedback_router import FeedbackRouter; FeedbackRouter('models/latency_predictor.pkl').run_chaos_test('latency_spike')"
```

See [advanced/README.md](advanced/README.md) for detailed documentation.

---

## 🤖 Bonus: Option 2 (Agent-Centric Approach)

As an additional exploration, I also implemented a **proof-of-concept LLM-based routing agent** in [`option2-agent/`](option2-agent/).

**Key Differences:**
- **Option 3 (This repo):** Fast ML predictions (~microseconds), predictive, black-box
- **Option 2 (Bonus):** LLM-based reasoning (~100ms), interpretable, explainable decisions

The agent approach excels when **explainability** is critical (regulatory, auditing) and decision logic changes frequently. See [option2-agent/README.md](option2-agent/README.md) for details and demo.

**Why both?** A hybrid system could use ML for 95% of fast requests and the agent for 5% of complex, high-value decisions requiring detailed reasoning.

