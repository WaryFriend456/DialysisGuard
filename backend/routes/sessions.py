"""
Session Routes — Start/stop/manage dialysis sessions.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from bson import ObjectId
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from models.schemas import SessionCreate, SessionResponse, SessionStatus
from routes.auth import get_current_user
from config import settings

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


def _session_response(session: dict, db) -> SessionResponse:
    current_step = int(session.get("current_step", len(session.get("time_series_data", []))))
    total_steps = int(session.get("total_steps", settings.SIMULATION_TIME_STEPS))
    status = session.get("status", SessionStatus.ACTIVE.value)
    return SessionResponse(
        id=str(session["_id"]),
        patient_id=session["patient_id"],
        started_by=session.get("started_by"),
        start_time=session["start_time"],
        end_time=session.get("end_time"),
        status=status,
        risk_profile=session.get("risk_profile"),
        current_step=current_step,
        total_steps=total_steps,
        can_resume=status in {SessionStatus.ACTIVE.value, SessionStatus.PAUSED.value} and current_step < total_steps,
        time_series_count=len(session.get("time_series_data", [])),
        prediction_count=len(session.get("predictions", [])),
        alert_count=db.alerts.count_documents({"session_id": str(session["_id"])})
    )


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

    existing = db.sessions.find_one(
        {
            "patient_id": data.patient_id,
            "status": {"$in": [SessionStatus.ACTIVE.value, SessionStatus.PAUSED.value]}
        },
        sort=[("start_time", -1)]
    )
    if existing:
        # If a new risk_profile was requested, update the existing session
        if data.risk_profile and data.risk_profile != existing.get("risk_profile"):
            db.sessions.update_one(
                {"_id": existing["_id"]},
                {"$set": {"risk_profile": data.risk_profile}}
            )
            existing["risk_profile"] = data.risk_profile
        return _session_response(existing, db)
    
    session_doc = {
        "patient_id": data.patient_id,
        "started_by": user["id"],
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "status": SessionStatus.ACTIVE.value,
        "risk_profile": data.risk_profile,
        "current_step": 0,
        "total_steps": settings.SIMULATION_TIME_STEPS,
        "explicit_stop": False,
        "time_series_data": [],
        "predictions": [],
        "explanations": [],
        "report": None
    }
    
    result = db.sessions.insert_one(session_doc)
    session_doc["_id"] = result.inserted_id
    return _session_response(session_doc, db)


@router.get("/stats")
async def get_session_stats(user=Depends(get_current_user)):
    """Get session statistics including active session count."""
    db = get_database()
    active_count = db.sessions.count_documents(
        {"status": {"$in": [SessionStatus.ACTIVE.value, SessionStatus.PAUSED.value]}}
    )
    total_count = db.sessions.count_documents({})
    completed_count = db.sessions.count_documents(
        {"status": {"$in": [SessionStatus.COMPLETED.value, SessionStatus.STOPPED.value]}}
    )
    return {
        "active_count": active_count,
        "total_count": total_count,
        "completed_count": completed_count,
    }


@router.get("/active/current")
async def get_current_active_session(
    patient_id: str | None = Query(default=None),
    user=Depends(get_current_user)
):
    """Get the most recent resumable session for the current user, optionally scoped to a patient."""
    db = get_database()
    query = {
        "status": {"$in": [SessionStatus.ACTIVE.value, SessionStatus.PAUSED.value]}
    }
    if patient_id:
        query["patient_id"] = patient_id
    else:
        query["started_by"] = user["id"]

    session = db.sessions.find_one(query, sort=[("start_time", -1)])
    if not session:
        return {"session": None}

    return {"session": _session_response(session, db)}


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

    return _session_response(session, db)


@router.post("/{session_id}/stop")
async def stop_session(session_id: str, user=Depends(get_current_user)):
    """Stop an active session and generate report."""
    db = get_database()
    
    session = db.sessions.find_one({"_id": ObjectId(session_id)})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["status"] not in {SessionStatus.ACTIVE.value, SessionStatus.PAUSED.value}:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Generate auto-report
    report = _generate_session_report(session, session_id)
    
    db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {
            "status": SessionStatus.COMPLETED.value,
            "end_time": datetime.utcnow().isoformat(),
            "explicit_stop": True,
            "current_step": len(session.get("time_series_data", [])),
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
        result.append(_session_response(s, db))
    
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
