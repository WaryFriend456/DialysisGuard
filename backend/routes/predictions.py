"""
Prediction Routes
"""
from fastapi import APIRouter, Depends
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.schemas import PredictionRequest, PredictionResponse
from routes.auth import get_current_user
from services.ml_service import ml_service
from services.xai_service import xai_service

router = APIRouter(prefix="/api/predict", tags=["Predictions"])


@router.post("/", response_model=PredictionResponse)
async def predict(data: PredictionRequest, user=Depends(get_current_user)):
    """Single prediction with uncertainty."""
    if data.sequence:
        X = ml_service.preprocess_sequence(
            [{k: v for k, v in zip(ml_service.feature_config['feature_names'], step)}
             for step in data.sequence]
        )
    else:
        X = ml_service.preprocess_sequence([data.patient_data])
    
    uncertainty = ml_service.predict_with_uncertainty(X)
    risk_prob = float(uncertainty['mean'])
    risk_level = ml_service.get_risk_level(risk_prob)
    
    return PredictionResponse(
        risk_probability=round(risk_prob, 4),
        risk_level=risk_level,
        confidence={
            "mean": round(float(uncertainty['mean']), 4),
            "lower": round(float(uncertainty['ci_lower']), 4),
            "upper": round(float(uncertainty['ci_upper']), 4),
            "std": round(float(uncertainty['std']), 4)
        },
        recommendations=ml_service.get_recommendations(risk_level)
    )


@router.post("/risk-assessment")
async def risk_assessment(data: PredictionRequest, user=Depends(get_current_user)):
    """Full risk assessment with XAI data."""
    if data.sequence:
        raw_data = [{k: v for k, v in zip(ml_service.feature_config['feature_names'], step)}
                    for step in data.sequence]
    else:
        raw_data = [data.patient_data]
    
    X = ml_service.preprocess_sequence(raw_data)
    
    # Prediction with uncertainty
    uncertainty = ml_service.predict_with_uncertainty(X)
    risk_prob = float(uncertainty['mean'])
    risk_level = ml_service.get_risk_level(risk_prob)
    
    # XAI data
    shap_data = xai_service.compute_shap_values(X)
    attention_data = xai_service.get_attention_weights(X)
    
    nl_explanation = xai_service.generate_nl_explanation(
        risk_prob, risk_level,
        shap_data.get('top_contributors', []),
        attention_data, uncertainty, [], "stable"
    )
    
    return {
        "prediction": {
            "risk_probability": round(risk_prob, 4),
            "risk_level": risk_level,
            "confidence": uncertainty
        },
        "explanations": {
            "shap": shap_data,
            "attention": attention_data,
            "nl_explanation": nl_explanation
        },
        "recommendations": ml_service.get_recommendations(
            risk_level, shap_data.get('top_contributors', [])
        )
    }
