"""
DialysisGuard Configuration
"""
import os
import sys
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "DialysisGuard API"
    DEBUG: bool = True
    
    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB: str = "dialysisguard"
    
    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dialysisguard-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # ML Model paths
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
    MODEL_PATH: str = os.path.join(MODEL_DIR, "gru_model.h5")
    SCALER_PATH: str = os.path.join(MODEL_DIR, "scaler.pkl")
    ENCODERS_PATH: str = os.path.join(MODEL_DIR, "label_encoders.pkl")
    FEATURE_CONFIG_PATH: str = os.path.join(MODEL_DIR, "feature_config.json")
    MODEL_CARD_PATH: str = os.path.join(MODEL_DIR, "model_card.json")
    
    # XAI
    SHAP_BACKGROUND_SAMPLES: int = 100
    MC_DROPOUT_PASSES: int = 20
    REALTIME_MC_DROPOUT_PASSES: int = 6
    ANALYSIS_MC_DROPOUT_PASSES: int = 20

    # Simulation
    SIMULATION_INTERVAL_SECONDS: float = 3.0
    SIMULATION_TIME_STEPS: int = 30

    # Runtime support
    SUPPORTED_PYTHON_MAJOR: int = 3
    SUPPORTED_PYTHON_MINOR: int = 11

    @property
    def supported_python_version(self) -> str:
        return f"{self.SUPPORTED_PYTHON_MAJOR}.{self.SUPPORTED_PYTHON_MINOR}.x"

    @property
    def current_python_version(self) -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    class Config:
        env_file = ".env"


settings = Settings()
