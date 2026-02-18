"""
Session Routes — Start/stop/manage dialysis sessions.
"""
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from models.schemas import SessionCreate, SessionResponse, SessionStatus
from routes.auth import get_current_user

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.post("/", response_model=SessionResponse)
async def create_session(data: SessionCreate, user=Depends(get_current_user)):
    """Start a new dialysis session for a patient."""
    db = get_database()
    
    # Verify patient exists
    try:
        patient = db.patients.find_one({"_id": ObjectId(data.patient_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID")
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Auto-close any stale active sessions for this patient
    db.sessions.update_many(
        {"patient_id": data.patient_id, "status": "active"},
        {"$set": {"status": "completed", "end_time": datetime.utcnow().isoformat()}}
    )
    
    session_doc = {
        "patient_id": data.patient_id,
        "started_by": user["id"],
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "status": "active",
        "risk_profile": data.risk_profile,
        "time_series_data": [],
        "predictions": [],
        "explanations": [],
        "report": None
    }
    
    result = db.sessions.insert_one(session_doc)
    
    return SessionResponse(
        id=str(result.inserted_id),
        patient_id=data.patient_id,
        started_by=user["id"],
        start_time=session_doc["start_time"],
        status=SessionStatus.ACTIVE
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, user=Depends(get_current_user)):
    """Get session details."""
    db = get_database()
    
    try:
        session = db.sessions.find_one({"_id": ObjectId(session_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=str(session["_id"]),
        patient_id=session["patient_id"],
        started_by=session.get("started_by"),
        start_time=session["start_time"],
        end_time=session.get("end_time"),
        status=session["status"],
        time_series_count=len(session.get("time_series_data", [])),
        prediction_count=len(session.get("predictions", [])),
        alert_count=db.alerts.count_documents({"session_id": session_id})
    )


@router.post("/{session_id}/stop")
async def stop_session(session_id: str, user=Depends(get_current_user)):
    """Stop an active session and generate report."""
    db = get_database()
    
    session = db.sessions.find_one({"_id": ObjectId(session_id)})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Generate auto-report
    report = _generate_session_report(session, session_id)
    
    db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {
            "status": "completed",
            "end_time": datetime.utcnow().isoformat(),
            "report": report
        }}
    )
    
    return {"message": "Session stopped", "report": report}


@router.get("/{session_id}/report")
async def get_session_report(session_id: str, user=Depends(get_current_user)):
    """Get auto-generated session report."""
    db = get_database()
    
    session = db.sessions.find_one({"_id": ObjectId(session_id)})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.get("report"):
        report = _generate_session_report(session, session_id)
        return report
    
    return session["report"]


@router.get("/patient/{patient_id}")
async def get_patient_sessions(patient_id: str, user=Depends(get_current_user)):
    """Get all sessions for a patient."""
    db = get_database()
    
    sessions = list(db.sessions.find(
        {"patient_id": patient_id}
    ).sort("start_time", -1))
    
    result = []
    for s in sessions:
        result.append(SessionResponse(
            id=str(s["_id"]),
            patient_id=s["patient_id"],
            started_by=s.get("started_by"),
            start_time=s["start_time"],
            end_time=s.get("end_time"),
            status=s["status"],
            time_series_count=len(s.get("time_series_data", [])),
            prediction_count=len(s.get("predictions", []))
        ))
    
    return {"sessions": result}


def _generate_session_report(session: dict, session_id: str) -> dict:
    """Auto-generate comprehensive session report."""
    db = get_database()
    
    predictions = session.get("predictions", [])
    time_series = session.get("time_series_data", [])
    alerts = list(db.alerts.find({"session_id": session_id}))
    
    # Calculate statistics
    risk_values = [p.get("risk_probability", 0) for p in predictions]
    
    report = {
        "session_id": session_id,
        "patient_id": session.get("patient_id"),
        "duration_steps": len(time_series),
        "start_time": session.get("start_time"),
        "summary": {
            "total_predictions": len(predictions),
            "avg_risk": round(sum(risk_values) / len(risk_values), 4) if risk_values else 0,
            "max_risk": round(max(risk_values), 4) if risk_values else 0,
            "min_risk": round(min(risk_values), 4) if risk_values else 0,
            "peak_risk_step": risk_values.index(max(risk_values)) if risk_values else 0,
        },
        "alerts": {
            "total": len(alerts),
            "critical": sum(1 for a in alerts if a.get("severity") == "CRITICAL"),
            "high": sum(1 for a in alerts if a.get("severity") == "HIGH"),
            "moderate": sum(1 for a in alerts if a.get("severity") == "MODERATE"),
            "acknowledged": sum(1 for a in alerts if a.get("acknowledged")),
        },
        "generated_at": datetime.utcnow().isoformat()
    }
    
    return report
