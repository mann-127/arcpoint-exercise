"""Feedback-aware router with closed-loop learning.

Combines:
- Base ML router for predictions
- Feedback loop for continuous improvement
- Anomaly detection for unusual patterns
- Drift detection for model degradation
"""
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced.feedback_loop import FeedbackCollector, OnlineLearner, DriftDetector, ABTestFramework
from advanced.anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FeedbackAwareRouter:
    """Router with closed-loop learning and anomaly detection."""
    
    LATENCY_THRESHOLD = 300
    
    def __init__(self, model_path: str = "models/latency_predictor.pkl"):
        """Initialize feedback-aware router.
        
        Args:
            model_path: Path to trained model
        """
        # Load base model
        if os.path.exists(model_path):
            self.base_model = joblib.load(model_path)
            logger.info(f"Loaded base model from {model_path}")
        else:
            self.base_model = None
            logger.warning("No base model found, using online learning only")
        
        # Initialize feedback components
        self.feedback = FeedbackCollector()
        self.online_learner = OnlineLearner()
        self.drift_detector = DriftDetector()
        self.anomaly_detector = AnomalyDetector(warmup_samples=20)
        self.ab_test = ABTestFramework()
        
        # State
        self.request_count = 0
        self.use_online_model = False
        
        logger.info("FeedbackAwareRouter initialized")
        
    def _compute_features(self, metrics: dict) -> list:
        """Extract features from metrics."""
        return [
            metrics.get('current_load', 100),
            metrics.get('latency_ma_5', metrics.get('avg_latency_ms', 100)),
            metrics.get('latency_slope', 0)
        ]
    
    def predict(self, features: list) -> float:
        """Make prediction using appropriate model.
        
        Args:
            features: Feature vector
            
        Returns:
            Predicted latency in ms
        """
        # A/B test: decide which model to use
        request_id = f"req_{self.request_count}"
        variant = self.ab_test.assign_variant(request_id)
        
        if variant == "treatment" and self.online_learner.is_fitted:
            # Use online model (treatment)
            return self.online_learner.predict(features)
        elif self.base_model:
            # Use base model (control)
            X = pd.DataFrame([features], columns=['current_load', 'latency_ma_5', 'latency_slope'])
            return self.base_model.predict(X)[0]
        else:
            # Fallback: simple heuristic
            return features[0] * 0.8 + 50  # load * 0.8 + base
    
    def route(self, metrics: dict) -> tuple:
        """Make routing decision with feedback.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Tuple of (decision_string, predicted_latency)
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        # Check for anomalies first
        anomaly = self.anomaly_detector.update(metrics)
        if anomaly and anomaly.severity == "high":
            logger.warning(f"Routing override due to anomaly: {anomaly.description}")
            return "🚨 EMERGENCY_REROUTE (Anomaly)", 999.0
        
        # Extract features and predict
        features = self._compute_features(metrics)
        predicted_latency = self.predict(features)
        
        # Make decision
        if predicted_latency > self.LATENCY_THRESHOLD:
            decision = f"⚠️ REROUTE (Pred: {predicted_latency:.0f}ms)"
        else:
            decision = f"✅ PRIMARY (Pred: {predicted_latency:.0f}ms)"
            
        return decision, predicted_latency
    
    def record_outcome(self, predicted: float, actual: float, decision: str) -> None:
        """Record outcome and update models.
        
        Args:
            predicted: Predicted latency
            actual: Actual observed latency
            decision: Routing decision made
        """
        request_id = f"req_{self.request_count}"
        features = [100, 100, 0]  # Placeholder, ideally store from route()
        
        # Record feedback
        record = self.feedback.record(
            request_id=request_id,
            features=features,
            predicted_latency=predicted,
            actual_latency=actual,
            routing_decision=decision,
            threshold=self.LATENCY_THRESHOLD
        )
        
        # Update online learner
        self.online_learner.partial_fit(features, actual)
        
        # Check for drift
        error = abs(predicted - actual)
        drift = self.drift_detector.update(error)
        
        if drift:
            logger.warning("Model drift detected! Consider retraining.")
            
        # Record for A/B test
        variant = self.ab_test.assign_variant(request_id)
        self.ab_test.record_outcome(variant, error)
    
    def get_status(self) -> dict:
        """Get router status."""
        return {
            "requests_processed": self.request_count,
            "feedback_metrics": self.feedback.get_metrics(),
            "drift_status": self.drift_detector.get_status(),
            "anomaly_status": self.anomaly_detector.get_status(),
            "ab_test_results": self.ab_test.get_results()
        }


def run_simulation():
    """Run a simulation demonstrating the feedback-aware router."""
    logger.info("🚀 Starting Feedback-Aware Router Simulation")
    
    router = FeedbackAwareRouter()
    
    print("\n" + "="*70)
    print("🔄 FEEDBACK-AWARE ROUTING SIMULATION")
    print("="*70)
    print(f"{'TIME':<8} | {'LOAD':<6} | {'ACTUAL':<10} | {'PREDICTED':<10} | {'DECISION':<30}")
    print("-"*70)
    
    # Simulate: Normal → Spike → Recovery
    loads = [100]*10 + [250]*15 + [100]*10
    
    for i, load in enumerate(loads):
        # Simulate actual latency (ground truth)
        base_latency = 50 + (load * 0.8)
        if load > 200:
            base_latency += (i * 15)  # Degradation over time
        actual_latency = base_latency + np.random.normal(0, 10)
        
        # Create metrics
        metrics = {
            'current_load': load,
            'avg_latency_ms': actual_latency,
            'latency_ma_5': actual_latency,
            'latency_slope': np.random.normal(0, 5) if load <= 200 else 30,
            'error_rate': 0.01 if load <= 200 else 0.05,
            'load_change_rate': 0 if i == 0 else loads[i] - loads[i-1]
        }
        
        # Get routing decision
        decision, predicted = router.route(metrics)
        
        # Record outcome (simulated feedback)
        router.record_outcome(predicted, actual_latency, decision)
        
        print(f"T+{i}m     | {load:<6} | {actual_latency:>6.1f} ms  | {predicted:>6.1f} ms  | {decision}")
        
        time.sleep(0.1)
    
    # Print final status
    print("\n" + "="*70)
    print("📊 FINAL STATUS")
    print("="*70)
    
    status = router.get_status()
    
    print(f"\n📈 Feedback Metrics:")
    print(f"   Total requests: {status['feedback_metrics']['total']}")
    print(f"   Decision accuracy: {status['feedback_metrics']['accuracy']*100:.1f}%")
    print(f"   MAE: {status['feedback_metrics']['mae']:.1f} ms")
    
    print(f"\n🔍 Drift Detection:")
    print(f"   Samples analyzed: {status['drift_status']['samples_seen']}")
    print(f"   Drift detected: {status['drift_status']['drift_detected']}")
    
    print(f"\n🚨 Anomaly Detection:")
    print(f"   Total anomalies: {status['anomaly_status']['total_anomalies']}")
    print(f"   High severity: {status['anomaly_status']['high_severity_count']}")
    
    print(f"\n🧪 A/B Test Results:")
    ab = status['ab_test_results']
    if ab.get('status') != 'insufficient_data':
        print(f"   Control MAE: {ab.get('control_mae', 0):.1f} ms")
        print(f"   Treatment MAE: {ab.get('treatment_mae', 0):.1f} ms")
        print(f"   Recommendation: {ab.get('recommendation', 'N/A')}")
    else:
        print("   Insufficient data for comparison")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_simulation()
