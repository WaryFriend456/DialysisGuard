"""
XAI / Explanation Routes
"""
from fastapi import APIRouter, Depends
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.schemas import (
    SHAPRequest, SHAPResponse, AttentionResponse,
    WhatIfRequest, WhatIfResponse,
    CounterfactualRequest, CounterfactualResponse,
    SensitivityResponse
)
from routes.auth import get_current_user
from services.ml_service import ml_service
from services.xai_service import xai_service

router = APIRouter(prefix="/api/explain", tags=["Explainable AI"])


@router.post("/shap")
async def get_shap(data: SHAPRequest, user=Depends(get_current_user)):
    """Get SHAP feature attributions for a prediction."""
    raw_data = [{k: v for k, v in zip(ml_service.feature_config['feature_names'], step)}
                for step in data.sequence]
    X = ml_service.preprocess_sequence(raw_data)
    return xai_service.compute_shap_values(X)


@router.post("/attention")
async def get_attention(data: SHAPRequest, user=Depends(get_current_user)):
    """Get temporal attention weights."""
    raw_data = [{k: v for k, v in zip(ml_service.feature_config['feature_names'], step)}
                for step in data.sequence]
    X = ml_service.preprocess_sequence(raw_data)
    return xai_service.get_attention_weights(X)


@router.post("/what-if")
async def what_if(data: WhatIfRequest, user=Depends(get_current_user)):
    """What-If analysis: modify parameters and see risk change."""
    raw_data = [data.patient_data]
    X = ml_service.preprocess_sequence(raw_data)
    return xai_service.what_if_analysis(X, raw_data, data.modifications)


@router.post("/counterfactual")
async def counterfactual(data: CounterfactualRequest, user=Depends(get_current_user)):
    """Find minimal changes to achieve target risk level."""
    raw_data = [data.patient_data]
    X = ml_service.preprocess_sequence(raw_data)
    return xai_service.find_counterfactual(X, raw_data, data.target_risk)


@router.post("/sensitivity")
async def sensitivity(data: SHAPRequest, user=Depends(get_current_user)):
    """Feature sensitivity analysis."""
    raw_data = [{k: v for k, v in zip(ml_service.feature_config['feature_names'], step)}
                for step in data.sequence]
    X = ml_service.preprocess_sequence(raw_data)
    return xai_service.sensitivity_analysis(X, raw_data)


@router.get("/model-card")
async def model_card(user=Depends(get_current_user)):
    """Get model transparency card."""
    return xai_service.get_model_card()
