"""
ML Service - model loading, preprocessing, inference, and uncertainty.
"""

import json
import logging
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


logger = logging.getLogger("dialysisguard.ml")


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

        self._validate_runtime()
        self._import_tensorflow()
        self._preflight_model_files()

        model_dir = settings.MODEL_DIR
        logger.info("Loading model artifacts from %s", model_dir)
        sys.setrecursionlimit(10000)

        self.model, self._attention_extractor = self._load_model(model_dir)

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        encoders_path = os.path.join(model_dir, "label_encoders.pkl")
        config_path = os.path.join(model_dir, "feature_config.json")

        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        with open(config_path, encoding="utf-8") as handle:
            self.feature_config = json.load(handle)

        self.feature_names = self.feature_config["feature_names"]
        self.n_timesteps = int(self.feature_config.get("n_timesteps", 30))
        self.n_features = int(self.feature_config.get("n_features", len(self.feature_names)))
        self.n_static = int(self.feature_config.get("n_static", 21))
        self.n_temporal = int(self.feature_config.get("n_temporal", 11))
        self.categorical_features = self.feature_config.get("categorical_features", [])
        self._oov_warning_cache = set()
        self._categorical_defaults = {
            "Gender": "Male",
            "Kidney Failure Cause": "Other",
            "Dialysate Composition": "Standard",
            "Vascular Access Type": "Fistula",
            "Dialyzer Type": "High-flux",
            "Disease Severity": "Moderate",
        }
        self._categorical_aliases = {
            "Gender": {
                "m": "Male",
                "male": "Male",
                "f": "Female",
                "female": "Female",
            },
            "Kidney Failure Cause": {
                "diabetes": "Diabetes",
                "hypertension": "Hypertension",
                "glomerulonephritis": "Other",
                "polycystic": "Other",
                "other": "Other",
            },
            "Dialysate Composition": {
                "standard": "Standard",
                "custom": "Customized",
                "customized": "Customized",
            },
            "Vascular Access Type": {
                "fistula": "Fistula",
                "graft": "Graft",
                "catheter": "Catheter",
            },
            "Dialyzer Type": {
                "high flux": "High-flux",
                "high-flux": "High-flux",
                "low flux": "Low-flux",
                "low-flux": "Low-flux",
            },
            "Disease Severity": {
                "mild": "Mild",
                "moderate": "Moderate",
                "severe": "Severe",
                "critical": "Severe",
            },
        }

        self._warmup_model()

        self._initialized = True
        logger.info(
            "ML service ready with %s features and %s timesteps",
            self.n_features,
            self.n_timesteps,
        )

    def _validate_runtime(self):
        supported = (
            settings.SUPPORTED_PYTHON_MAJOR,
            settings.SUPPORTED_PYTHON_MINOR,
        )
        current = (sys.version_info.major, sys.version_info.minor)
        if current != supported:
            raise RuntimeError(
                "Unsupported Python runtime "
                f"{settings.current_python_version}. "
                f"Use Python {settings.supported_python_version}."
            )

    def _import_tensorflow(self):
        try:
            import tensorflow as tf
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow could not be imported. "
                "Install backend requirements in a Python 3.11 environment."
            ) from exc
        self.tf = tf

    def _preflight_model_files(self):
        model_dir = settings.MODEL_DIR
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"Model directory not found: {model_dir}")

        required_files = [
            "model_architecture.json",
            "model_weights.weights.h5",
            "scaler.pkl",
            "label_encoders.pkl",
            "feature_config.json",
        ]
        missing = [
            os.path.join(model_dir, filename)
            for filename in required_files
            if not os.path.exists(os.path.join(model_dir, filename))
        ]
        if missing:
            raise RuntimeError(
                "Missing required model artifacts:\n- " + "\n- ".join(missing)
            )

        try:
            with open(os.path.join(model_dir, "model_architecture.json"), encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            raise RuntimeError("Model architecture file is unreadable or invalid JSON.") from exc

        try:
            with open(os.path.join(model_dir, "feature_config.json"), encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            raise RuntimeError("Feature config file is unreadable or invalid JSON.") from exc

    def _load_model(self, model_dir):
        arch_path = os.path.join(model_dir, "model_architecture.json")
        weights_path = os.path.join(model_dir, "model_weights.weights.h5")

        if os.path.exists(arch_path) and os.path.exists(weights_path):
            try:
                from ml.attention_gru import AttentionGRUModel

                with open(arch_path, encoding="utf-8") as handle:
                    arch = json.load(handle)

                builder = AttentionGRUModel(
                    n_timesteps=arch["n_timesteps"],
                    n_features=arch["n_features"],
                    n_static=arch["n_static"],
                    n_temporal=arch["n_temporal"],
                    gru_units=tuple(arch["gru_units"]),
                    attention_units=arch["attention_units"],
                    dense_units=tuple(arch["dense_units"]),
                    dropout_rate=arch["dropout_rate"],
                    learning_rate=arch["learning_rate"],
                )
                builder.build()
                builder.model.load_weights(weights_path)
                logger.info("Loaded model from architecture + weights")
                return builder.model, builder.attention_model
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load model weights from {weights_path}. "
                    "The saved artifacts may be incompatible or corrupt."
                ) from exc

        try:
            from ml.attention_gru import BahdanauAttention

            model_path = os.path.join(model_dir, "gru_model.keras")
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, "gru_model.h5")
            model = self.tf.keras.models.load_model(
                model_path,
                custom_objects={"BahdanauAttention": BahdanauAttention},
            )
            logger.info("Loaded fallback serialized model from %s", model_path)
            return model, None
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the fallback serialized model. "
                "Check that the saved model artifacts are present and not corrupt."
            ) from exc

    def _warmup_model(self):
        warmup_input = np.zeros((1, self.n_timesteps, self.n_features), dtype=np.float32)
        self.model(warmup_input, training=False)
        if self._attention_extractor is not None:
            self._attention_extractor(warmup_input, training=False)
        logger.info("Model warm-up completed")

    def _warn_oov_category(self, feature: str, raw_value, mapped_value):
        key = (feature, str(raw_value), str(mapped_value))
        if key in self._oov_warning_cache:
            return
        self._oov_warning_cache.add(key)
        logger.warning(
            "OOV category for %s: %r -> %r",
            feature,
            raw_value,
            mapped_value,
        )

    def _default_category_index(self, feature: str, encoder) -> int:
        default_name = self._categorical_defaults.get(feature)
        classes = list(getattr(encoder, "classes_", []))
        if default_name in classes:
            return int(encoder.transform([default_name])[0])
        return 0

    def _canonicalize_category(self, feature: str, value: str, encoder):
        classes = list(getattr(encoder, "classes_", []))
        normalized = value.strip().lower().replace("_", " ").replace("-", " ")
        for item in classes:
            item_norm = str(item).strip().lower().replace("_", " ").replace("-", " ")
            if item_norm == normalized:
                return str(item)
        alias_map = self._categorical_aliases.get(feature, {})
        return alias_map.get(normalized)

    def preprocess_sequence(self, raw_data: list) -> np.ndarray:
        """Convert raw simulation data to scaled model input."""
        if not self._initialized:
            self.initialize()
        if not raw_data:
            raise ValueError("Cannot preprocess an empty sequence.")

        sequence = []
        for step in raw_data:
            row = []
            for feat in self.feature_names:
                val = step.get(feat, 0.0)
                if feat in self.categorical_features:
                    encoder = self.label_encoders[feat]
                    if isinstance(val, str):
                        canonical = self._canonicalize_category(feat, val, encoder)
                        if canonical and canonical in encoder.classes_:
                            if canonical != val:
                                self._warn_oov_category(feat, val, canonical)
                            val = int(encoder.transform([canonical])[0])
                        elif val in encoder.classes_:
                            val = int(encoder.transform([val])[0])
                        else:
                            fallback_idx = self._default_category_index(feat, encoder)
                            fallback_val = encoder.inverse_transform([fallback_idx])[0]
                            self._warn_oov_category(feat, val, fallback_val)
                            val = fallback_idx
                    else:
                        if isinstance(val, (int, float)):
                            candidate = int(val)
                            if 0 <= candidate < len(encoder.classes_):
                                val = candidate
                            else:
                                fallback_idx = self._default_category_index(feat, encoder)
                                fallback_val = encoder.inverse_transform([fallback_idx])[0]
                                self._warn_oov_category(feat, val, fallback_val)
                                val = fallback_idx
                        else:
                            fallback_idx = self._default_category_index(feat, encoder)
                            fallback_val = encoder.inverse_transform([fallback_idx])[0]
                            self._warn_oov_category(feat, val, fallback_val)
                            val = fallback_idx
                else:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.0
                row.append(val)
            sequence.append(row)

        X = np.array(sequence, dtype=np.float32)

        if len(sequence) < self.n_timesteps:
            pad_length = self.n_timesteps - len(sequence)
            first_step = X[0:1, :]
            padding = np.repeat(first_step, pad_length, axis=0)
            X = np.concatenate([padding, X], axis=0)
        elif len(sequence) > self.n_timesteps:
            X = X[-self.n_timesteps :, :]

        X_flat = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(1, self.n_timesteps, self.n_features)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled.astype(np.float32)

    def predict(self, X: np.ndarray) -> dict:
        """Make a standard prediction."""
        if not self._initialized:
            self.initialize()

        pred = self.model(X, training=False).numpy()
        prob = float(pred.squeeze())
        risk_level = self.get_risk_level(prob)

        return {
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "risk_category": self._risk_category(prob),
            "recommendation": self._recommendation(risk_level),
        }

    def predict_with_uncertainty(self, X: np.ndarray, n_passes: int | None = None) -> dict:
        """MC Dropout prediction with vectorized uncertainty estimation."""
        if not self._initialized:
            self.initialize()

        passes = int(n_passes or settings.ANALYSIS_MC_DROPOUT_PASSES)
        repeated = np.repeat(X, passes, axis=0)
        predictions = self.model(repeated, training=True).numpy().reshape(passes, -1)

        mean = float(np.mean(predictions))
        std = float(np.std(predictions))
        ci_lower = float(max(0.0, mean - 1.96 * std))
        ci_upper = float(min(1.0, mean + 1.96 * std))

        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "confidence": round(max(0.0, min(1.0, 1.0 - std)), 4),
        }

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Extract attention weights from the model."""
        if not self._initialized:
            self.initialize()

        if self._attention_extractor is None:
            try:
                attention_layer = self.model.get_layer("attention")
                dropout_output = self.model.get_layer("dropout_2").output
                score = attention_layer.V(self.tf.nn.tanh(attention_layer.W(dropout_output)))
                attn_weights = self.tf.nn.softmax(score, axis=1)
                self._attention_extractor = self.tf.keras.Model(
                    inputs=self.model.input,
                    outputs=attn_weights,
                )
            except Exception as exc:
                logger.warning("Falling back to uniform attention weights: %s", exc)
                self._attention_extractor = None

        if self._attention_extractor is not None:
            try:
                weights = self._attention_extractor(X, training=False).numpy()
                return weights
            except Exception as exc:
                logger.warning("Attention extraction failed, using uniform weights: %s", exc)

        return np.ones((1, self.n_timesteps, 1), dtype=np.float32) / self.n_timesteps

    @staticmethod
    def get_risk_level(probability: float) -> str:
        if probability < 0.25:
            return "low"
        if probability < 0.50:
            return "moderate"
        if probability < 0.75:
            return "high"
        return "critical"

    @staticmethod
    def _risk_category(probability: float) -> str:
        if probability < 0.25:
            return "Patient appears hemodynamically stable"
        if probability < 0.50:
            return "Mild risk indicators present - monitor closely"
        if probability < 0.75:
            return "Significant instability predicted - prepare intervention"
        return "Critical instability imminent - immediate action required"

    @staticmethod
    def _recommendation(risk_level: str) -> str:
        recommendations = {
            "low": "Continue standard monitoring protocol",
            "moderate": "Increase monitoring frequency; check fluid removal rate",
            "high": "Consider reducing ultrafiltration rate; prepare vasopressors",
            "critical": "Stop ultrafiltration; administer saline bolus; call attending physician",
        }
        return recommendations.get(risk_level, "Assess patient status")

    def get_recommendations(self, risk_level: str, top_features: list = None) -> list:
        recs = [self._recommendation(risk_level)]
        if top_features:
            feature_advice = {
                "Current_BP": "Monitor blood pressure closely",
                "Current_HR": "Check heart rate and rhythm",
                "BP_Change": "Assess rate of blood pressure decline",
                "HR_Change": "Evaluate heart rate variability",
                "Fluid Removal Rate (ml/hour)": "Consider adjusting ultrafiltration rate",
                "Time_Minutes": "Note elapsed session time",
            }
            for feat in (top_features or [])[:3]:
                name = feat.get("name", "") if isinstance(feat, dict) else str(feat)
                direction = feat.get("direction") if isinstance(feat, dict) else None
                if name in feature_advice and direction == "risk_increasing":
                    recs.append(feature_advice[name])
        return recs


ml_service = MLService()
