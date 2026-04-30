"""
DialysisGuard backend preflight checks.

Usage:
    python scripts/preflight.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
from pymongo import MongoClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, ok=True, detail=detail)
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=str(exc))


def check_python_version() -> str:
    current = (sys.version_info.major, sys.version_info.minor)
    expected = (settings.SUPPORTED_PYTHON_MAJOR, settings.SUPPORTED_PYTHON_MINOR)
    if current != expected:
        raise RuntimeError(
            f"Expected Python {settings.supported_python_version}, got {settings.current_python_version}."
        )
    return f"Python {settings.current_python_version}"


def check_tensorflow_import() -> str:
    import tensorflow as tf  # noqa: WPS433

    return f"TensorFlow {tf.__version__} import OK"


def check_model_artifacts() -> str:
    model_dir = Path(settings.MODEL_DIR)
    required = [
        "model_architecture.json",
        "model_weights.weights.h5",
        "scaler.pkl",
        "label_encoders.pkl",
        "feature_config.json",
        "model_card.json",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing model artifacts: {', '.join(missing)}")

    for name in ["model_architecture.json", "feature_config.json", "model_card.json"]:
        with (model_dir / name).open(encoding="utf-8") as handle:
            json.load(handle)

    for name in ["model_weights.weights.h5", "scaler.pkl", "label_encoders.pkl"]:
        path = model_dir / name
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Artifact is empty: {path}")

    return f"Artifacts found in {model_dir}"


def check_encoder_config_consistency() -> str:
    model_dir = Path(settings.MODEL_DIR)
    feature_cfg = json.loads((model_dir / "feature_config.json").read_text(encoding="utf-8"))
    arch_cfg = json.loads((model_dir / "model_architecture.json").read_text(encoding="utf-8"))
    encoders = joblib.load(model_dir / "label_encoders.pkl")

    feature_names = feature_cfg.get("feature_names", [])
    n_features = int(feature_cfg.get("n_features", 0))
    n_static = int(feature_cfg.get("n_static", 0))
    n_temporal = int(feature_cfg.get("n_temporal", 0))
    categorical = set(feature_cfg.get("categorical_features", []))

    if len(feature_names) != n_features:
        raise RuntimeError(
            f"feature_config mismatch: len(feature_names)={len(feature_names)} vs n_features={n_features}"
        )
    if n_static + n_temporal != n_features:
        raise RuntimeError(
            f"feature_config mismatch: n_static+n_temporal={n_static + n_temporal} vs n_features={n_features}"
        )
    if int(arch_cfg.get("n_features", -1)) != n_features:
        raise RuntimeError(
            f"architecture mismatch: model_architecture n_features={arch_cfg.get('n_features')} vs feature_config n_features={n_features}"
        )
    if int(arch_cfg.get("n_timesteps", -1)) != int(feature_cfg.get("n_timesteps", -1)):
        raise RuntimeError("architecture mismatch: n_timesteps differs between architecture and feature config")

    missing_encoders = sorted(categorical - set(encoders.keys()))
    if missing_encoders:
        raise RuntimeError(f"Missing label encoders for categorical features: {missing_encoders}")

    # These are known frontend/API categories; they are remapped at runtime if unseen.
    app_categories = {
        "Disease Severity": {"Mild", "Moderate", "Severe", "Critical"},
        "Kidney Failure Cause": {"Diabetes", "Hypertension", "Glomerulonephritis", "Polycystic", "Other"},
    }
    gaps = []
    for feature, values in app_categories.items():
        if feature not in encoders:
            continue
        encoder_classes = {str(value) for value in encoders[feature].classes_}
        unseen = sorted(values - encoder_classes)
        if unseen:
            gaps.append(f"{feature}: {', '.join(unseen)}")

    if gaps:
        return "Encoder/config consistency OK (runtime mapping used for: " + " | ".join(gaps) + ")"
    return "Encoder/config consistency OK"


def check_model_load_path() -> str:
    from services.ml_service import ml_service  # noqa: WPS433

    ml_service.initialize()
    return f"Model initialized ({ml_service.n_features} features, {ml_service.n_timesteps} timesteps)"


def check_mongodb_connectivity() -> str:
    timeout_ms = int(os.getenv("PREFLIGHT_MONGO_TIMEOUT_MS", "3000"))
    client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=timeout_ms)
    try:
        client.admin.command("ping")
    finally:
        client.close()
    return f"MongoDB ping OK ({settings.MONGODB_URI})"


def main() -> int:
    checks = [
        ("Python Version", check_python_version),
        ("TensorFlow Import", check_tensorflow_import),
        ("Model Artifacts", check_model_artifacts),
        ("Encoder/Config Consistency", check_encoder_config_consistency),
        ("Model Initialization", check_model_load_path),
        ("MongoDB Connectivity", check_mongodb_connectivity),
    ]

    print("DialysisGuard Backend Preflight")
    print("=" * 36)
    results = [run_check(name, fn) for name, fn in checks]

    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        print(f"[{mark}] {result.name}: {result.detail}")

    failed = [item for item in results if not item.ok]
    if failed:
        print("\nPreflight failed. Fix the checks above before starting the backend.")
        return 1

    print("\nPreflight passed. Backend is ready to start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
