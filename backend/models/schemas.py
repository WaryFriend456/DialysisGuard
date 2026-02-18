"""
Pydantic Schemas for request/response validation.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ========================
# Enums
# ========================

class UserRole(str, Enum):
    DOCTOR = "doctor"
    CAREGIVER = "caregiver"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    STOPPED = "stopped"


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskTrend(str, Enum):
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"


# ========================
# Auth Schemas
# ========================

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=6)
    role: UserRole


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: UserRole
    created_at: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ========================
# Patient Schemas
# ========================

class PatientCreate(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str
    weight: float = Field(..., gt=30, lt=200)
    diabetes: bool = False
    hypertension: bool = False
    kidney_failure_cause: str = "Other"
    creatinine: float = Field(default=5.0, ge=0.5, le=15.0)
    urea: float = Field(default=50.0, ge=10, le=200)
    potassium: float = Field(default=4.5, ge=2.5, le=7.0)
    hemoglobin: float = Field(default=11.0, ge=5.0, le=18.0)
    hematocrit: float = Field(default=33.0, ge=15, le=55)
    albumin: float = Field(default=3.8, ge=1.5, le=5.5)
    dialysis_duration: float = Field(default=4.0, ge=2.0, le=8.0)
    dialysis_frequency: int = Field(default=3, ge=1, le=7)
    dialysate_composition: str = "Standard"
    vascular_access_type: str = "Fistula"
    dialyzer_type: str = "High-flux"
    urine_output: float = Field(default=500, ge=0, le=2000)
    dry_weight: float = Field(default=70.0, gt=30, lt=200)
    fluid_removal_rate: float = Field(default=350, ge=100, le=600)
    disease_severity: str = "Moderate"
    name: Optional[str] = None  # Display name for the patient


class PatientUpdate(BaseModel):
    age: Optional[int] = None
    weight: Optional[float] = None
    creatinine: Optional[float] = None
    urea: Optional[float] = None
    potassium: Optional[float] = None
    hemoglobin: Optional[float] = None
    hematocrit: Optional[float] = None
    albumin: Optional[float] = None
    dialysis_duration: Optional[float] = None
    dialysis_frequency: Optional[int] = None
    fluid_removal_rate: Optional[float] = None
    disease_severity: Optional[str] = None
    name: Optional[str] = None


class PatientResponse(BaseModel):
    id: str
    name: Optional[str] = None
    age: int
    gender: str
    weight: float
    diabetes: bool
    hypertension: bool
    kidney_failure_cause: str
    creatinine: float
    urea: float
    potassium: float
    hemoglobin: float
    hematocrit: float
    albumin: float
    dialysis_duration: float
    dialysis_frequency: int
    dialysate_composition: str
    vascular_access_type: str
    dialyzer_type: str
    urine_output: float
    dry_weight: float
    fluid_removal_rate: float
    disease_severity: str
    created_by: Optional[str] = None
    created_at: Optional[str] = None


# ========================
# Session Schemas
# ========================

class SessionCreate(BaseModel):
    patient_id: str
    risk_profile: Optional[str] = None  # "low", "moderate", "high", "critical"


class SessionResponse(BaseModel):
    id: str
    patient_id: str
    started_by: Optional[str] = None
    start_time: str
    end_time: Optional[str] = None
    status: SessionStatus
    time_series_count: int = 0
    prediction_count: int = 0
    alert_count: int = 0


# ========================
# Prediction Schemas
# ========================

class PredictionRequest(BaseModel):
    patient_data: Dict[str, Any]
    sequence: Optional[List[List[float]]] = None


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    confidence: Dict[str, float]
    risk_trend: Optional[str] = None
    recommendations: List[str] = []


# ========================
# Alert Schemas
# ========================

class AlertResponse(BaseModel):
    id: str
    session_id: str
    patient_id: str
    severity: AlertSeverity
    risk_probability: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    message: str
    nl_explanation: Optional[str] = None
    top_features: List[Dict] = []
    recommendations: List[str] = []
    acknowledged: bool = False
    escalation_level: int = 0
    created_at: str
    acknowledged_at: Optional[str] = None


# ========================
# XAI Schemas
# ========================

class SHAPRequest(BaseModel):
    sequence: List[List[float]]


class SHAPResponse(BaseModel):
    base_value: float
    shap_values: List[List[float]]
    feature_names: List[str]
    top_contributors: List[Dict]


class AttentionResponse(BaseModel):
    attention_weights: List[float]
    time_labels: List[str]


class WhatIfRequest(BaseModel):
    patient_data: Dict[str, Any]
    modifications: Dict[str, float]


class WhatIfResponse(BaseModel):
    original_risk: float
    modified_risk: float
    risk_delta: float
    confidence: Dict[str, float]
    explanation: str


class CounterfactualRequest(BaseModel):
    patient_data: Dict[str, Any]
    target_risk: float = 0.3


class CounterfactualResponse(BaseModel):
    suggestions: List[Dict]
    achievable: bool
    target_risk: float
    best_achievable_risk: float


class SensitivityResponse(BaseModel):
    sensitivities: List[Dict]


# ========================
# WebSocket Schemas (for documentation)
# ========================

class MonitoringPayload(BaseModel):
    time_minutes: int
    step: int
    vitals: Dict[str, float]
    prediction: Dict[str, Any]
    xai: Dict[str, Any]
    anomalies: List[Dict]
    alert: Optional[Dict] = None
