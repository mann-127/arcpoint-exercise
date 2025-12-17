"""Intelligent routing engine with predictive circuit breaker.

Simulates real-time routing decisions based on ML-predicted latency.
"""
import logging
import os
import time
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RealTimeFeatureStore:
    """In-memory feature store with sliding window aggregations."""
    
    WINDOW_SIZE = 10
    COLD_START_THRESHOLD = 5
    
    def __init__(self):
        self.buffer = pd.DataFrame(columns=['timestamp', 'current_load', 'avg_latency_ms'])
        
    def ingest(self, timestamp, load, latency):
        """Append new metric and maintain sliding window."""
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'current_load': load,
            'avg_latency_ms': latency
        }])
        
        if self.buffer.empty:
            self.buffer = new_row
        else:
            self.buffer = pd.concat([self.buffer, new_row], ignore_index=True)
        
        if len(self.buffer) > self.WINDOW_SIZE:
            self.buffer = self.buffer.iloc[-self.WINDOW_SIZE:]
            
    def get_features(self):
        """Compute features for model inference.
        
        Returns:
            list: [current_load, latency_ma_5, latency_slope] or None if cold start
        """
        if len(self.buffer) < self.COLD_START_THRESHOLD:
            return None
            
        current_load = self.buffer.iloc[-1]['current_load']
        latency_ma_5 = self.buffer['avg_latency_ms'].rolling(window=5).mean().iloc[-1]
        latency_slope = self.buffer['avg_latency_ms'].diff().iloc[-1]
        
        if pd.isna(latency_slope):
            latency_slope = 0.0
            
        return [current_load, latency_ma_5, latency_slope]


class IntelligentRouter:
    """ML-powered router with predictive circuit breaker."""
    
    LATENCY_THRESHOLD_MS = 300
    
    def __init__(self, model_path):
        """Load trained model.
        
        Args:
            model_path: Path to serialized model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run 'python src/model.py' first."
            )
        
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def decide(self, features):
        """Make routing decision based on predicted latency.
        
        Args:
            features: Feature vector or None (cold start)
            
        Returns:
            tuple: (decision_string, predicted_latency)
        """
        if features is None:
            return "ROUND_ROBIN (Cold Start)", 0.0
            
        X = pd.DataFrame(
            [features], 
            columns=['current_load', 'latency_ma_5', 'latency_slope']
        )
        predicted_latency = self.model.predict(X)[0]
        
        if predicted_latency > self.LATENCY_THRESHOLD_MS:
            decision = f"⚠️ REROUTE (Pred: {predicted_latency:.0f}ms)"
        else:
            decision = f"✅ PRIMARY (Pred: {predicted_latency:.0f}ms)"
            
        return decision, predicted_latency


def simulate_live_traffic():
    """Simulate traffic spike scenario and demonstrate routing decisions."""
    logger.info("Booting Arcpoint Context Engine...")
    
    store = RealTimeFeatureStore()
    try:
        router = IntelligentRouter("models/latency_predictor.pkl")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info("Connected to stream. Listening for metrics...")
    print("-" * 60)
    print(f"{'TIME':<10} | {'LOAD':<6} | {'ACTUAL LATENCY':<15} | {'DECISION':<25}")
    print("-" * 60)

    # Scenario: Normal → Spike → Recovery
    loads = [100]*5 + [250]*10 + [100]*5
    
    for i, load in enumerate(loads):
        base_latency = 50 + (load * 0.8)
        
        if load > 200:
            base_latency += (i * 20)
            
        current_latency = base_latency + np.random.normal(0, 5)
        
        store.ingest(i, load, current_latency)
        features = store.get_features()
        decision, pred_val = router.decide(features)
        
        print(f"T+{i}m       | {load:<6} | {current_latency:>6.1f} ms        | {decision}")
        time.sleep(0.2)


if __name__ == "__main__":
    simulate_live_traffic()
