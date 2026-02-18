"""
Alert Routes
"""
from fastapi import APIRouter, Depends, Query
from bson import ObjectId
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from routes.auth import get_current_user
from services.alert_service import alert_service

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("/")
async def list_alerts(
    severity: str = Query(None),
    acknowledged: bool = Query(None),
    session_id: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user=Depends(get_current_user)
):
    """List alerts with filters."""
    db = get_database()
    query = {}
    
    if severity:
        query["severity"] = severity
    if acknowledged is not None:
        query["acknowledged"] = acknowledged
    if session_id:
        query["session_id"] = session_id
    
    alerts = list(db.alerts.find(query).sort("created_at", -1).limit(limit))
    
    for a in alerts:
        a["id"] = str(a["_id"])
        del a["_id"]
    
    return {"alerts": alerts, "total": len(alerts)}


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user=Depends(get_current_user)):
    """Acknowledge an alert."""
    result = alert_service.acknowledge_alert(alert_id)
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"message": "Alert acknowledged", "alert_id": alert_id}


@router.get("/stats")
async def alert_stats(user=Depends(get_current_user)):
    """Get alert statistics."""
    return alert_service.get_alert_stats()
