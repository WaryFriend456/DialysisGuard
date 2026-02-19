"""
ML Service — Model Inference for Real-Time Monitoring

Handles loading the trained BiGRU model and performing inference
on streaming data from the simulation engine.

Key responsibilities:
    - Load model, scaler, label encoders from saved artifacts
    - Preprocess raw simulation data into model input format
    - Make predictions with uncertainty (MC Dropout)
    - Extract attention weights for XAI
    - Map risk probabilities to clinical risk levels

v2 changes:
    - Feature-repeat padding instead of zero-padding for partial sequences
    - Uses feature_config.json for n_static / n_temporal counts
    - Clinical risk thresholds refined
"""

import os
import sys
import json
import numpy as np
import joblib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class MLService:
    """Singleton service for hemodialysis instability prediction."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Load model and preprocessing artifacts from disk."""
        if self._initialized:
            return
        
        model_dir = settings.MODEL_DIR
        print(f"[ML Service] Loading model from: {model_dir}")
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Load model: prefer weights + architecture config (avoids serialization issues)
        arch_path = os.path.join(model_dir, 'model_architecture.json')
        weights_path = os.path.join(model_dir, 'model_weights.weights.h5')
        
        if os.path.exists(arch_path) and os.path.exists(weights_path):
            from ml.attention_gru import AttentionGRUModel
            with open(arch_path) as f:
                arch = json.load(f)
            
            builder = AttentionGRUModel(
                n_timesteps=arch['n_timesteps'],
                n_features=arch['n_features'],
                n_static=arch['n_static'],
                n_temporal=arch['n_temporal'],
                gru_units=tuple(arch['gru_units']),
                attention_units=arch['attention_units'],
                dense_units=tuple(arch['dense_units']),
                dropout_rate=arch['dropout_rate'],
                learning_rate=arch['learning_rate']
            )
            builder.build()
            builder.model.load_weights(weights_path)
            self.model = builder.model
            self._attention_extractor = builder.attention_model
            print(f"  Model loaded from architecture + weights")
        else:
            # Fallback to full model file
            from ml.attention_gru import BahdanauAttention
            model_path = os.path.join(model_dir, 'gru_model.keras')
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, 'gru_model.h5')
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'BahdanauAttention': BahdanauAttention}
            )
            self._attention_extractor = None
            print(f"  Model loaded: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        print(f"  Scaler loaded: {scaler_path}")
        
        # Load label encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        self.label_encoders = joblib.load(encoders_path)
        print(f"  Label encoders loaded: {encoders_path}")
        
        # Load feature config
        config_path = os.path.join(model_dir, 'feature_config.json')
        with open(config_path) as f:
            self.feature_config = json.load(f)
        
        self.feature_names = self.feature_config['feature_names']
        self.n_timesteps = self.feature_config.get('n_timesteps', 30)
        self.n_features = self.feature_config.get('n_features', len(self.feature_names))
        self.n_static = self.feature_config.get('n_static', 21)
        self.n_temporal = self.feature_config.get('n_temporal', 11)
        self.categorical_features = self.feature_config.get('categorical_features', [])
        print(f"  Features: {self.n_features} ({self.n_static} static + {self.n_temporal} temporal)")
        print(f"  Sequence length: {self.n_timesteps}")
        
        self._initialized = True
        print("[ML Service] ✅ Ready")
    
    def preprocess_sequence(self, raw_data: list) -> np.ndarray:
        """
        Convert raw simulation data to scaled model input.
        
        Args:
            raw_data: List of dicts, each representing one time step
                      from the simulation engine.
        
        Returns:
            X: numpy array of shape (1, n_timesteps, n_features)
               Padded with feature-repeat if fewer than n_timesteps steps.
        """
        if not self._initialized:
            self.initialize()
        
        feature_names = self.feature_names
        categorical_features = self.categorical_features
        
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
        
        X = np.array(sequence, dtype=np.float32)
        
        # Handle padding: if fewer than n_timesteps, pad with FIRST STEP
        # (not zeros — zero-padding is out-of-distribution for our scaler)
        if len(sequence) < self.n_timesteps:
            pad_length = self.n_timesteps - len(sequence)
            first_step = X[0:1, :]  # (1, n_features)
            padding = np.repeat(first_step, pad_length, axis=0)
            X = np.concatenate([padding, X], axis=0)
        elif len(sequence) > self.n_timesteps:
            # Take last n_timesteps
            X = X[-self.n_timesteps:, :]
        
        # Scale using the training-fitted scaler
        X_flat = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(1, self.n_timesteps, self.n_features)
        
        # Clean up any NaN/Inf
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_scaled.astype(np.float32)
    
    def predict(self, X: np.ndarray) -> dict:
        """
        Make a standard prediction.
        
        Args:
            X: Preprocessed input (1, n_timesteps, n_features)
        
        Returns:
            dict with probability, risk_level, risk_category
        """
        if not self._initialized:
            self.initialize()
        
        # Model predicts P(unstable) for the sequence
        pred = self.model.predict(X, verbose=0)
        prob = float(pred.squeeze())
        
        risk_level = self.get_risk_level(prob)
        
        return {
            'probability': round(prob, 4),
            'risk_level': risk_level,
            'risk_category': self._risk_category(prob),
            'recommendation': self._recommendation(risk_level)
        }
    
    def predict_with_uncertainty(self, X: np.ndarray, n_passes: int = 20) -> dict:
        """
        MC Dropout prediction with uncertainty estimation.
        
        Runs the model multiple times with dropout enabled.
        """
        if not self._initialized:
            self.initialize()
        
        predictions = []
        for _ in range(n_passes):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # (n_passes, 1, 1)
        
        mean = float(np.mean(predictions))
        std = float(np.std(predictions))
        ci_lower = float(max(0, mean - 1.96 * std))
        ci_upper = float(min(1, mean + 1.96 * std))
        
        return {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'confidence': round(1.0 - std, 4)  # Higher = more confident
        }
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights from the model.
        
        Returns:
            Attention weights (1, n_timesteps, 1) showing which
            time steps the model focused on.
        """
        if not self._initialized:
            self.initialize()
        
        if not hasattr(self, '_attention_extractor') or self._attention_extractor is None:
            try:
                # The BahdanauAttention layer produces (context, weights)
                # Find the attention layer's output in the model's graph
                attention_layer = self.model.get_layer('attention')
                # Get the attention layer's output node — it has 2 outputs
                # We need the second one (weights)
                # Rebuild a model that outputs attention weights
                dropout_2 = self.model.get_layer('dropout_2')
                
                # Re-compute attention using the layer's weights
                score = attention_layer.V(tf.nn.tanh(attention_layer.W(dropout_2.output)))
                attn_weights = tf.nn.softmax(score, axis=1)
                
                self._attention_extractor = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=attn_weights
                )
            except Exception as e:
                print(f"[ML Service] Warning: Building attention extractor failed: {e}")
                print(f"[ML Service] Falling back to uniform attention weights")
                self._attention_extractor = None
        
        if self._attention_extractor is not None:
            try:
                return self._attention_extractor.predict(X, verbose=0)
            except Exception as e:
                print(f"[ML Service] Warning: Attention extraction failed: {e}")
        
        return np.ones((1, self.n_timesteps, 1)) / self.n_timesteps
    
    @staticmethod
    def get_risk_level(probability: float) -> str:
        """Map probability to clinical risk level."""
        if probability < 0.25:
            return 'low'
        elif probability < 0.50:
            return 'moderate'
        elif probability < 0.75:
            return 'high'
        else:
            return 'critical'
    
    @staticmethod
    def _risk_category(probability: float) -> str:
        """Get descriptive risk category."""
        if probability < 0.25:
            return 'Patient appears hemodynamically stable'
        elif probability < 0.50:
            return 'Mild risk indicators present — monitor closely'
        elif probability < 0.75:
            return 'Significant instability predicted — prepare intervention'
        else:
            return 'Critical instability imminent — immediate action required'
    
    @staticmethod
    def _recommendation(risk_level: str) -> str:
        """Get clinical recommendation based on risk level."""
        recommendations = {
            'low': 'Continue standard monitoring protocol',
            'moderate': 'Increase monitoring frequency; check fluid removal rate',
            'high': 'Consider reducing ultrafiltration rate; prepare vasopressors',
            'critical': 'Stop ultrafiltration; administer saline bolus; call attending physician'
        }
        return recommendations.get(risk_level, 'Assess patient status')

    def get_recommendations(self, risk_level: str, top_features: list = None) -> list:
        """
        Generate clinical recommendations based on risk level and top features.
        Called by realtime.py during alert creation.
        """
        recs = [self._recommendation(risk_level)]

        if top_features:
            feature_advice = {
                'Current_BP': 'Monitor blood pressure closely',
                'Current_HR': 'Check heart rate and rhythm',
                'BP_Change': 'Assess rate of blood pressure decline',
                'HR_Change': 'Evaluate heart rate variability',
                'Fluid Removal Rate (ml/hour)': 'Consider adjusting ultrafiltration rate',
                'UFR_to_Weight': 'Review fluid removal relative to patient weight',
                'Session Time (mins)': 'Evaluate session timing',
                'Time_Minutes': 'Note elapsed session time',
            }
            for feat in (top_features or [])[:3]:
                name = feat.get('name', '') if isinstance(feat, dict) else str(feat)
                if name in feature_advice and feat.get('direction') == 'risk_increasing':
                    recs.append(feature_advice[name])

        return recs


# Singleton instance
ml_service = MLService()
