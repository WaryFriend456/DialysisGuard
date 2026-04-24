"""
Explainable AI service for monitoring and on-demand analysis.
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


logger = logging.getLogger("dialysisguard.xai")


class XAIService:
    """Provides explainability features for model predictions."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, ml_service):
        if self._initialized:
            return

        self.ml_service = ml_service
        self.tf = ml_service.tf
        self.feature_config = ml_service.feature_config
        self.feature_names = self.feature_config["feature_names"]
        with open(settings.MODEL_CARD_PATH, "r", encoding="utf-8") as handle:
            self.model_card = json.load(handle)

        self._shap_explainer = None
        self._shap_background = None
        self._initialized = True
        logger.info("XAI service initialized")

    def _get_shap_explainer(self, background_data=None):
        if self._shap_explainer is None:
            try:
                import shap

                if background_data is None:
                    n_bg = min(settings.SHAP_BACKGROUND_SAMPLES, 50)
                    n_ts = self.feature_config["n_timesteps"]
                    n_feat = self.feature_config["n_features"]
                    self._shap_background = np.zeros((n_bg, n_ts, n_feat), dtype=np.float32)
                else:
                    self._shap_background = background_data[: settings.SHAP_BACKGROUND_SAMPLES]

                self._shap_explainer = shap.DeepExplainer(
                    self.ml_service.model,
                    self._shap_background,
                )
                logger.info("SHAP explainer initialized")
            except Exception as exc:
                logger.warning("SHAP initialization failed, using fallback attribution: %s", exc)
                return None
        return self._shap_explainer

    def compute_shap_values(self, X: np.ndarray) -> dict:
        explainer = self._get_shap_explainer()
        if explainer is not None:
            try:
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                avg_shap = np.mean(shap_values[0], axis=0)
                base_value = float(explainer.expected_value)
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = float(base_value[0])

                contributions = self._build_contributions(avg_shap)
                return {
                    "base_value": round(base_value, 4),
                    "shap_values": shap_values[0].tolist(),
                    "feature_names": self.feature_names,
                    "top_contributors": contributions[:10],
                }
            except Exception as exc:
                logger.warning("SHAP computation failed, using fallback attribution: %s", exc)

        fallback = self.approximate_feature_importance_fast(X)
        return {
            "base_value": fallback["base_value"],
            "shap_values": [],
            "feature_names": self.feature_names,
            "top_contributors": fallback["top_contributors"],
        }

    def approximate_feature_importance_fast(self, X: np.ndarray) -> dict:
        """Fast gradient-based attribution suitable for realtime monitoring."""
        tensor = self.tf.convert_to_tensor(X)
        with self.tf.GradientTape() as tape:
            tape.watch(tensor)
            preds = self.ml_service.model(tensor, training=False)
        grads = tape.gradient(preds, tensor).numpy()[0]
        input_values = X[0]
        scores = np.mean(np.abs(grads * input_values), axis=0)
        contributions = self._build_contributions(np.mean(grads * input_values, axis=0), scores=scores)
        base_pred = self.ml_service.predict(X)["probability"]
        return {
            "base_value": round(float(base_pred), 4),
            "feature_names": self.feature_names,
            "top_contributors": contributions[:10],
        }

    def _approximate_feature_importance(self, X: np.ndarray) -> dict:
        """Compatibility fallback used by existing code paths."""
        return self.approximate_feature_importance_fast(X)

    def _build_contributions(self, signed_values, scores=None):
        contributions = []
        for idx, feat in enumerate(self.feature_names):
            value = float(signed_values[idx])
            abs_value = float(scores[idx]) if scores is not None else abs(value)
            contributions.append(
                {
                    "name": feat,
                    "value": round(value, 4),
                    "direction": "risk_increasing" if value > 0 else "risk_decreasing",
                    "abs_value": round(abs_value, 4),
                }
            )
        contributions.sort(key=lambda item: item["abs_value"], reverse=True)
        return contributions

    def get_attention_weights(self, X: np.ndarray) -> dict:
        weights = self.ml_service.get_attention_weights(X)
        weights_flat = weights.squeeze().tolist()
        return {
            "attention_weights": weights_flat,
            "time_labels": [f"{i * 8} min" for i in range(len(weights_flat))],
            "peak_attention_step": int(np.argmax(weights_flat)),
            "peak_attention_time": f"{int(np.argmax(weights_flat)) * 8} min",
        }

    def what_if_analysis(self, X: np.ndarray, raw_data: list, modifications: dict) -> dict:
        original_pred = self.ml_service.predict(X)["probability"]
        original_uncertainty = self.ml_service.predict_with_uncertainty(
            X, n_passes=settings.ANALYSIS_MC_DROPOUT_PASSES
        )

        modified_data = [step.copy() for step in raw_data]
        for step in modified_data:
            for key, value in modifications.items():
                if key in step:
                    step[key] = value

        X_modified = self.ml_service.preprocess_sequence(modified_data)
        modified_pred = self.ml_service.predict(X_modified)["probability"]
        modified_uncertainty = self.ml_service.predict_with_uncertainty(
            X_modified, n_passes=settings.ANALYSIS_MC_DROPOUT_PASSES
        )

        return {
            "original_risk": round(float(original_pred), 4),
            "modified_risk": round(float(modified_pred), 4),
            "risk_delta": round(float(modified_pred - original_pred), 4),
            "confidence": {
                "original": {
                    "mean": round(float(original_uncertainty["mean"]), 4),
                    "lower": round(float(original_uncertainty["ci_lower"]), 4),
                    "upper": round(float(original_uncertainty["ci_upper"]), 4),
                },
                "modified": {
                    "mean": round(float(modified_uncertainty["mean"]), 4),
                    "lower": round(float(modified_uncertainty["ci_lower"]), 4),
                    "upper": round(float(modified_uncertainty["ci_upper"]), 4),
                },
            },
            "explanation": self._generate_whatif_explanation(
                original_pred, modified_pred, modifications
            ),
        }

    def find_counterfactual(self, X: np.ndarray, raw_data: list, target_risk: float = 0.3) -> dict:
        original_pred = self.ml_service.predict(X)["probability"]
        modifiable = {
            "Fluid Removal Rate (ml/hour)": {"min": 100, "max": 600, "step": -25},
            "Current_BP": {"min": 80, "max": 180, "step": 5},
            "Current_HR": {"min": 50, "max": 120, "step": -3},
        }

        suggestions = []
        best_risk = original_pred
        for feat, params in modifiable.items():
            current_val = raw_data[-1].get(feat, 0) if raw_data else 0
            for delta_mult in range(1, 6):
                new_val = current_val + params["step"] * delta_mult
                new_val = max(params["min"], min(params["max"], new_val))
                if new_val == current_val:
                    continue
                modified_data = [step.copy() for step in raw_data]
                for step in modified_data:
                    if feat in step:
                        step[feat] = new_val
                X_mod = self.ml_service.preprocess_sequence(modified_data)
                new_risk = self.ml_service.predict(X_mod)["probability"]
                if new_risk < original_pred:
                    suggestions.append(
                        {
                            "feature": feat,
                            "current_value": round(float(current_val), 1),
                            "suggested_value": round(float(new_val), 1),
                            "change": round(float(new_val - current_val), 1),
                            "resulting_risk": round(float(new_risk), 4),
                            "risk_reduction": round(float(original_pred - new_risk), 4),
                        }
                    )
                    best_risk = min(best_risk, new_risk)
                    break

        suggestions.sort(key=lambda item: item["risk_reduction"], reverse=True)
        return {
            "suggestions": suggestions[:5],
            "achievable": best_risk <= target_risk,
            "target_risk": round(target_risk, 4),
            "best_achievable_risk": round(float(best_risk), 4),
            "original_risk": round(float(original_pred), 4),
        }

    def sensitivity_analysis(self, X: np.ndarray, raw_data: list) -> dict:
        original_pred = self.ml_service.predict(X)["probability"]
        sensitivities = []
        numerical_features = [
            f
            for f in self.feature_names
            if f not in self.feature_config.get("categorical_features", [])
            and f not in self.feature_config.get("boolean_features", [])
        ]

        for feat in numerical_features[:15]:
            current_val = raw_data[-1].get(feat, 0) if raw_data else 0
            if current_val == 0:
                continue

            up_data = [step.copy() for step in raw_data]
            for step in up_data:
                if feat in step:
                    step[feat] = step[feat] * 1.1
            X_up = self.ml_service.preprocess_sequence(up_data)
            up_risk = self.ml_service.predict(X_up)["probability"]

            down_data = [step.copy() for step in raw_data]
            for step in down_data:
                if feat in step:
                    step[feat] = step[feat] * 0.9
            X_down = self.ml_service.preprocess_sequence(down_data)
            down_risk = self.ml_service.predict(X_down)["probability"]

            sensitivities.append(
                {
                    "feature": feat,
                    "current_value": round(float(current_val), 2),
                    "risk_at_minus_10pct": round(float(down_risk), 4),
                    "risk_at_current": round(float(original_pred), 4),
                    "risk_at_plus_10pct": round(float(up_risk), 4),
                    "sensitivity": round(float(abs(up_risk - down_risk)), 4),
                }
            )

        sensitivities.sort(key=lambda item: item["sensitivity"], reverse=True)
        return {"sensitivities": sensitivities}

    def generate_nl_explanation(
        self,
        risk_prob: float,
        risk_level: str,
        top_features: list,
        attention_data: dict,
        uncertainty: dict,
        anomalies: list,
        trend: str = "stable",
    ) -> str:
        risk_label = str(risk_level).upper()
        explanation = f"Risk is {risk_label} ({risk_prob:.0%})"

        ci_lower = uncertainty.get("ci_lower", risk_prob)
        ci_upper = uncertainty.get("ci_upper", risk_prob)
        ci_width = ci_upper - ci_lower
        if ci_width > 0.15:
            explanation += f" with notable uncertainty (95% CI: {ci_lower:.0%}-{ci_upper:.0%})."
        else:
            explanation += f" with high confidence (95% CI: {ci_lower:.0%}-{ci_upper:.0%})."

        if top_features:
            increasing = [f for f in top_features[:5] if f.get("direction") == "risk_increasing"]
            if increasing:
                factors = ", ".join(
                    feature["name"].replace("_", " ") for feature in increasing[:3]
                )
                explanation += f" Main drivers: {factors}."

        if attention_data and attention_data.get("peak_attention_time"):
            explanation += (
                f" The model focused most on data from {attention_data['peak_attention_time']} into the session."
            )

        anomaly_types = {item.get("type", "") for item in anomalies or []}
        if "hypotension" in anomaly_types:
            explanation += " Hypotension detected."
        if "tachycardia" in anomaly_types:
            explanation += " Tachycardia detected."
        if "rapid_decline" in anomaly_types:
            explanation += " Rapid vital decline observed."

        if trend == "increasing":
            explanation += " Risk trend is rising and warrants close attention."
        elif trend == "decreasing":
            explanation += " Risk trend is improving."

        return explanation.strip()

    def calculate_risk_trend(self, recent_predictions: list) -> str:
        if len(recent_predictions) < 3:
            return "stable"

        last_5 = recent_predictions[-5:]
        x = np.arange(len(last_5))
        y = np.array(last_5)
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.02:
            return "increasing"
        if slope < -0.02:
            return "decreasing"
        return "stable"

    def forecast_risk(self, current_pred: float, trend: str, n_steps: int = 5) -> list:
        trend_rates = {"increasing": 0.03, "stable": 0.0, "decreasing": -0.02}
        rate = trend_rates.get(trend, 0.0)
        forecast = []
        risk = current_pred
        for _ in range(n_steps):
            risk = risk + rate + np.random.normal(0, 0.01)
            risk = max(0.0, min(1.0, risk))
            forecast.append(round(float(risk), 4))
        return forecast

    def get_model_card(self) -> dict:
        return self.model_card

    def _generate_whatif_explanation(self, original: float, modified: float, modifications: dict) -> str:
        changes = ", ".join(f"{key.replace('_', ' ')} to {val}" for key, val in modifications.items())
        delta = modified - original
        direction = "increases" if delta > 0 else "decreases"
        return (
            f"Modifying {changes} {direction} the predicted risk "
            f"from {original:.1%} to {modified:.1%} (change: {delta:+.1%})."
        )


xai_service = XAIService()
