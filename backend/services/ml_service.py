"""
ML Service — Model inference, MC Dropout, attention extraction.
Loads the trained GRU model and provides prediction utilities.
"""
import os
import sys
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml'))
from attention_gru import AttentionGRUModel, BahdanauAttention


class MLService:
    """Manages model loading, prediction, and uncertainty estimation."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Load model and preprocessing artifacts."""
        if self._initialized:
            return
        
        print("🧠 Loading ML model and artifacts...")
        
        # Load feature config
        with open(settings.FEATURE_CONFIG_PATH, 'r') as f:
            self.feature_config = json.load(f)
        
        # Load scaler
        self.scaler = joblib.load(settings.SCALER_PATH)
        
        # Load label encoders
        self.label_encoders = joblib.load(settings.ENCODERS_PATH)
        
        # Load model card
        with open(settings.MODEL_CARD_PATH, 'r') as f:
            self.model_card = json.load(f)
        
        # Load model with custom objects
        self.model = tf.keras.models.load_model(
            settings.MODEL_PATH,
            custom_objects={'BahdanauAttention': BahdanauAttention}
        )
        
        # Build the model builder wrapper for MC Dropout & attention
        n_timesteps = self.feature_config['n_timesteps']
        n_features = self.feature_config['n_features']
        
        self.builder = AttentionGRUModel(n_timesteps, n_features)
        self.builder.model = self.model
        self.builder._build_attention_model()
        
        self._initialized = True
        print(f"✅ Model loaded: {n_features} features, {n_timesteps} timesteps")
    
    def preprocess_sequence(self, raw_data: list) -> np.ndarray:
        """
        Preprocess a sequence of raw vital readings.
        raw_data: list of dicts, each containing feature values for one time step.
        Returns: scaled numpy array (1, timesteps, features)
        """
        feature_names = self.feature_config['feature_names']
        n_timesteps = self.feature_config['n_timesteps']
        
        # Categorical features that need label encoding
        categorical_features = set(self.label_encoders.keys())
        
        # Convert to array, encoding categoricals
        sequence = []
        for step in raw_data:
            row = []
            for feat in feature_names:
                val = step.get(feat, 0.0)
                if feat in categorical_features:
                    # Encode using label encoder
                    encoder = self.label_encoders[feat]
                    if isinstance(val, str):
                        if val in encoder.classes_:
                            val = int(encoder.transform([val])[0])
                        else:
                            val = 0  # fallback for unknown category
                    else:
                        val = int(val) if isinstance(val, (int, float)) else 0
                else:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.0
                row.append(val)
            sequence.append(row)
        
        # Pad with zeros if fewer than n_timesteps
        while len(sequence) < n_timesteps:
            sequence.insert(0, [0.0] * len(feature_names))
        
        # Take only last n_timesteps
        sequence = sequence[-n_timesteps:]
        
        X = np.array([sequence], dtype=np.float32)
        
        # Scale
        n_ts, n_feat = X.shape[1], X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = self.scaler.transform(X_flat)
        X = X_scaled.reshape(1, n_ts, n_feat).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def predict(self, X: np.ndarray) -> float:
        """Standard prediction — returns risk probability."""
        pred = self.model.predict(X, verbose=0)
        return float(pred.squeeze())
    
    def predict_with_uncertainty(self, X: np.ndarray, n_passes: int = None) -> dict:
        """MC Dropout prediction with confidence interval."""
        n = n_passes or settings.MC_DROPOUT_PASSES
        return self.builder.predict_with_dropout(X, n_passes=n)
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Extract attention weights for explainability."""
        return self.builder.get_attention_weights(X)
    
    def get_risk_level(self, probability: float) -> str:
        """Classify risk level from probability."""
        if probability >= 0.75:
            return "CRITICAL"
        elif probability >= 0.50:
            return "HIGH"
        elif probability >= 0.30:
            return "MODERATE"
        return "LOW"
    
    def get_recommendations(self, risk_level: str, top_features: list = None) -> list:
        """Generate clinical recommendations based on risk level and contributing features."""
        recs = []
        
        if risk_level in ("HIGH", "CRITICAL"):
            recs.append("Assess patient vitals immediately")
            recs.append("Consider reducing ultrafiltration rate")
            recs.append("Prepare for potential fluid bolus or intervention")
        
        if risk_level == "CRITICAL":
            recs.append("Alert attending physician immediately")
            recs.append("Consider early session termination if patient deteriorates")
        
        if top_features:
            for feat in top_features[:3]:
                name = feat.get('name', '')
                direction = feat.get('direction', '')
                if 'BP' in name and direction == 'risk_increasing':
                    recs.append("Monitor blood pressure closely — contributing to elevated risk")
                elif 'HR' in name and direction == 'risk_increasing':
                    recs.append("Monitor heart rate — compensatory tachycardia detected")
                elif 'Fluid' in name:
                    recs.append("Consider adjusting fluid removal rate")
        
        if risk_level == "MODERATE":
            recs.append("Continue monitoring — increased vigilance recommended")
        elif risk_level == "LOW":
            recs.append("Continue standard monitoring")
        
        return recs[:5]


# Singleton instance
ml_service = MLService()
