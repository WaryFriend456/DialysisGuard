"""
Explainable AI (XAI) Service — SHAP, What-If, Counterfactual, NL Explanations.
Provides the 7 XAI pillars for transparent predictions.
"""
import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class XAIService:
    """Provides explainability features for model predictions."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, ml_service):
        """Initialize with reference to ML service."""
        if self._initialized:
            return
            
        self.ml_service = ml_service
        self.feature_config = ml_service.feature_config
        self.feature_names = self.feature_config['feature_names']
        
        # Load model card
        with open(settings.MODEL_CARD_PATH, 'r') as f:
            self.model_card = json.load(f)
        
        # SHAP explainer will be initialized on first use (lazy loading)
        self._shap_explainer = None
        self._shap_background = None
        
        self._initialized = True
        print("✅ XAI Service initialized")
    
    def _get_shap_explainer(self, background_data=None):
        """Lazy-initialize SHAP DeepExplainer."""
        if self._shap_explainer is None:
            try:
                import shap
                
                if background_data is None:
                    # Generate synthetic background data
                    n_bg = min(settings.SHAP_BACKGROUND_SAMPLES, 50)
                    n_ts = self.feature_config['n_timesteps']
                    n_feat = self.feature_config['n_features']
                    self._shap_background = np.zeros((n_bg, n_ts, n_feat), dtype=np.float32)
                else:
                    self._shap_background = background_data[:settings.SHAP_BACKGROUND_SAMPLES]
                
                self._shap_explainer = shap.DeepExplainer(
                    self.ml_service.model,
                    self._shap_background
                )
                print("✅ SHAP DeepExplainer initialized")
            except Exception as e:
                print(f"⚠️ SHAP initialization failed: {e}")
                return None
        
        return self._shap_explainer
    
    def compute_shap_values(self, X: np.ndarray) -> dict:
        """
        Compute SHAP values for a prediction.
        
        Returns feature attributions showing how each feature
        pushes the prediction above or below the base rate.
        """
        explainer = self._get_shap_explainer()
        
        if explainer is not None:
            try:
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # shap_values shape: (1, timesteps, features)
                # Average across timesteps for per-feature attribution
                avg_shap = np.mean(shap_values[0], axis=0)  # (features,)
                
                # Get base value
                base_value = float(explainer.expected_value)
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = float(base_value[0])
                
                # Top contributors
                contributions = []
                for i, feat in enumerate(self.feature_names):
                    contributions.append({
                        'name': feat,
                        'value': round(float(avg_shap[i]), 4),
                        'direction': 'risk_increasing' if avg_shap[i] > 0 else 'risk_decreasing',
                        'abs_value': round(abs(float(avg_shap[i])), 4)
                    })
                
                contributions.sort(key=lambda x: x['abs_value'], reverse=True)
                
                return {
                    'base_value': round(base_value, 4),
                    'shap_values': shap_values[0].tolist(),
                    'feature_names': self.feature_names,
                    'top_contributors': contributions[:10]
                }
            except Exception as e:
                print(f"SHAP computation error: {e}")
        
        # Fallback: use gradient-based approximation
        return self._approximate_feature_importance(X)
    
    def _approximate_feature_importance(self, X: np.ndarray) -> dict:
        """
        Approximate feature importance using prediction perturbation.
        Fallback when SHAP is unavailable.
        """
        base_pred = self.ml_service.predict(X)
        importances = []
        
        for i, feat in enumerate(self.feature_names):
            X_perturbed = X.copy()
            X_perturbed[0, :, i] = 0.0  # Zero out feature
            perturbed_pred = self.ml_service.predict(X_perturbed)
            importance = base_pred - perturbed_pred
            
            importances.append({
                'name': feat,
                'value': round(float(importance), 4),
                'direction': 'risk_increasing' if importance > 0 else 'risk_decreasing',
                'abs_value': round(abs(float(importance)), 4)
            })
        
        importances.sort(key=lambda x: x['abs_value'], reverse=True)
        
        return {
            'base_value': round(float(base_pred), 4),
            'shap_values': [],
            'feature_names': self.feature_names,
            'top_contributors': importances[:10]
        }
    
    def get_attention_weights(self, X: np.ndarray) -> dict:
        """
        Get temporal attention weights from the model.
        Shows which time steps the model focused on.
        """
        weights = self.ml_service.get_attention_weights(X)
        weights_flat = weights.squeeze().tolist()
        
        # Generate time labels
        time_labels = [f"{i * 8} min" for i in range(len(weights_flat))]
        
        return {
            'attention_weights': weights_flat,
            'time_labels': time_labels,
            'peak_attention_step': int(np.argmax(weights_flat)),
            'peak_attention_time': f"{int(np.argmax(weights_flat)) * 8} min"
        }
    
    def what_if_analysis(self, X: np.ndarray, raw_data: list,
                          modifications: dict) -> dict:
        """
        What-If analysis — re-predict with modified parameters.
        
        Modify specific features and see how the prediction changes.
        """
        original_pred = self.ml_service.predict(X)
        original_uncertainty = self.ml_service.predict_with_uncertainty(X)
        
        # Apply modifications to raw data
        modified_data = [step.copy() for step in raw_data]
        for step in modified_data:
            for key, value in modifications.items():
                if key in step:
                    step[key] = value
        
        # Re-preprocess and predict
        X_modified = self.ml_service.preprocess_sequence(modified_data)
        modified_pred = self.ml_service.predict(X_modified)
        modified_uncertainty = self.ml_service.predict_with_uncertainty(X_modified)
        
        risk_delta = modified_pred - original_pred
        
        explanation = self._generate_whatif_explanation(
            original_pred, modified_pred, modifications
        )
        
        return {
            'original_risk': round(float(original_pred), 4),
            'modified_risk': round(float(modified_pred), 4),
            'risk_delta': round(float(risk_delta), 4),
            'confidence': {
                'original': {
                    'mean': round(float(original_uncertainty['mean']), 4),
                    'lower': round(float(original_uncertainty['ci_lower']), 4),
                    'upper': round(float(original_uncertainty['ci_upper']), 4),
                },
                'modified': {
                    'mean': round(float(modified_uncertainty['mean']), 4),
                    'lower': round(float(modified_uncertainty['ci_lower']), 4),
                    'upper': round(float(modified_uncertainty['ci_upper']), 4),
                }
            },
            'explanation': explanation
        }
    
    def find_counterfactual(self, X: np.ndarray, raw_data: list,
                             target_risk: float = 0.3) -> dict:
        """
        Find minimal parameter changes to achieve target risk level.
        
        Searches for small modifications that bring risk below target.
        """
        original_pred = self.ml_service.predict(X)
        
        # Define modifiable features with their ranges
        modifiable = {
            'Fluid Removal Rate (ml/hour)': {'min': 100, 'max': 600, 'step': -25},
            'Current_BP': {'min': 80, 'max': 180, 'step': 5},
            'Current_HR': {'min': 50, 'max': 120, 'step': -3},
        }
        
        suggestions = []
        best_risk = original_pred
        
        for feat, params in modifiable.items():
            # Try different values
            current_val = raw_data[-1].get(feat, 0) if raw_data else 0
            
            for delta_mult in range(1, 6):
                new_val = current_val + params['step'] * delta_mult
                new_val = max(params['min'], min(params['max'], new_val))
                
                if new_val == current_val:
                    continue
                
                # Test this modification
                modified_data = [step.copy() for step in raw_data]
                for step in modified_data:
                    if feat in step:
                        step[feat] = new_val
                
                X_mod = self.ml_service.preprocess_sequence(modified_data)
                new_risk = self.ml_service.predict(X_mod)
                
                if new_risk < original_pred:
                    suggestions.append({
                        'feature': feat,
                        'current_value': round(float(current_val), 1),
                        'suggested_value': round(float(new_val), 1),
                        'change': round(float(new_val - current_val), 1),
                        'resulting_risk': round(float(new_risk), 4),
                        'risk_reduction': round(float(original_pred - new_risk), 4)
                    })
                    
                    if new_risk < best_risk:
                        best_risk = new_risk
                    
                    break  # Found a useful suggestion for this feature
        
        suggestions.sort(key=lambda x: x['risk_reduction'], reverse=True)
        
        return {
            'suggestions': suggestions[:5],
            'achievable': best_risk <= target_risk,
            'target_risk': round(target_risk, 4),
            'best_achievable_risk': round(float(best_risk), 4),
            'original_risk': round(float(original_pred), 4)
        }
    
    def sensitivity_analysis(self, X: np.ndarray, raw_data: list) -> dict:
        """
        Sensitivity analysis — how much does each feature affect the prediction?
        Tests ±10% changes in each numerical feature.
        """
        original_pred = self.ml_service.predict(X)
        sensitivities = []
        
        numerical_features = [f for f in self.feature_names
                              if f not in self.feature_config.get('categorical_features', [])
                              and f not in self.feature_config.get('boolean_features', [])]
        
        for feat in numerical_features[:15]:  # Limit to top 15
            current_val = raw_data[-1].get(feat, 0) if raw_data else 0
            if current_val == 0:
                continue
            
            # Test +10%
            up_data = [step.copy() for step in raw_data]
            for step in up_data:
                if feat in step:
                    step[feat] = step[feat] * 1.1
            X_up = self.ml_service.preprocess_sequence(up_data)
            up_risk = self.ml_service.predict(X_up)
            
            # Test -10%
            down_data = [step.copy() for step in raw_data]
            for step in down_data:
                if feat in step:
                    step[feat] = step[feat] * 0.9
            X_down = self.ml_service.preprocess_sequence(down_data)
            down_risk = self.ml_service.predict(X_down)
            
            sensitivities.append({
                'feature': feat,
                'current_value': round(float(current_val), 2),
                'risk_at_minus_10pct': round(float(down_risk), 4),
                'risk_at_current': round(float(original_pred), 4),
                'risk_at_plus_10pct': round(float(up_risk), 4),
                'sensitivity': round(float(abs(up_risk - down_risk)), 4)
            })
        
        sensitivities.sort(key=lambda x: x['sensitivity'], reverse=True)
        
        return {'sensitivities': sensitivities}
    
    def generate_nl_explanation(self, risk_prob: float, risk_level: str,
                                 top_features: list, attention_data: dict,
                                 uncertainty: dict, anomalies: list,
                                 trend: str = "stable") -> str:
        """
        Generate a natural language explanation combining all XAI signals.
        
        Produces a human-readable paragraph explaining WHY the risk is
        at its current level, designed for clinical professionals.
        """
        # Risk level descriptions
        level_desc = {
            "LOW": "low",
            "MODERATE": "moderately elevated",
            "HIGH": "elevated",
            "CRITICAL": "critically elevated"
        }
        
        risk_desc = level_desc.get(risk_level, "uncertain")
        
        # Start with overall risk assessment
        explanation = f"Risk is {risk_level} ({risk_prob:.0%})"
        
        # Add confidence
        ci_lower = uncertainty.get('ci_lower', risk_prob)
        ci_upper = uncertainty.get('ci_upper', risk_prob)
        ci_width = ci_upper - ci_lower
        
        if ci_width > 0.15:
            explanation += f" with notable uncertainty (95% CI: {ci_lower:.0%}–{ci_upper:.0%})"
        else:
            explanation += f" with high confidence (95% CI: {ci_lower:.0%}–{ci_upper:.0%})"
        
        explanation += "."
        
        # Add primary contributing factors
        if top_features:
            increasing = [f for f in top_features[:5] if f.get('direction') == 'risk_increasing']
            decreasing = [f for f in top_features[:5] if f.get('direction') == 'risk_decreasing']
            
            if increasing:
                factors = ", ".join([f['name'].replace('_', ' ') for f in increasing[:3]])
                explanation += f" Primary risk drivers: {factors}."
            
            if decreasing:
                factors = ", ".join([f['name'].replace('_', ' ') for f in decreasing[:2]])
                explanation += f" Protective factors: {factors}."
        
        # Add temporal focus from attention
        if attention_data and attention_data.get('peak_attention_time'):
            explanation += f" The model focused most on data from {attention_data['peak_attention_time']} into the session."
        
        # Add anomalies
        if anomalies:
            anom_types = set(a.get('type', '') for a in anomalies)
            if 'hypotension' in anom_types:
                explanation += " ⚠️ Hypotension detected."
            if 'tachycardia' in anom_types:
                explanation += " ⚠️ Tachycardia detected."
            if 'rapid_decline' in anom_types:
                explanation += " ⚠️ Rapid vital sign decline observed."
        
        # Add trend
        trend_desc = {
            "increasing": "Risk is trending upward — continued monitoring critical.",
            "decreasing": "Risk trend is improving.",
            "stable": ""
        }
        if trend in trend_desc and trend_desc[trend]:
            explanation += f" {trend_desc[trend]}"
        
        return explanation
    
    def calculate_risk_trend(self, recent_predictions: list) -> str:
        """
        Calculate risk trend from recent predictions.
        Returns: "increasing", "stable", or "decreasing"
        """
        if len(recent_predictions) < 3:
            return "stable"
        
        last_5 = recent_predictions[-5:]
        if len(last_5) < 3:
            last_5 = recent_predictions
        
        # Linear trend
        x = np.arange(len(last_5))
        y = np.array(last_5)
        
        if len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.02:
                return "increasing"
            elif slope < -0.02:
                return "decreasing"
        
        return "stable"
    
    def forecast_risk(self, current_pred: float, trend: str,
                       n_steps: int = 5) -> list:
        """
        Simple predictive forecast for next N time steps.
        Projects current risk based on trend.
        """
        trend_rates = {
            "increasing": 0.03,
            "stable": 0.0,
            "decreasing": -0.02
        }
        
        rate = trend_rates.get(trend, 0.0)
        forecast = []
        risk = current_pred
        
        for i in range(n_steps):
            risk = risk + rate + np.random.normal(0, 0.01)
            risk = max(0.0, min(1.0, risk))
            forecast.append(round(float(risk), 4))
        
        return forecast
    
    def get_model_card(self) -> dict:
        """Return the model transparency card."""
        return self.model_card
    
    def _generate_whatif_explanation(self, original: float, modified: float,
                                     modifications: dict) -> str:
        """Generate explanation for what-if analysis result."""
        changes = []
        for key, val in modifications.items():
            changes.append(f"{key.replace('_', ' ')} to {val}")
        
        change_str = ", ".join(changes)
        delta = modified - original
        direction = "increases" if delta > 0 else "decreases"
        
        return (f"Modifying {change_str} {direction} the predicted risk "
                f"from {original:.1%} to {modified:.1%} "
                f"(change: {delta:+.1%}).")


# Singleton
xai_service = XAIService()
