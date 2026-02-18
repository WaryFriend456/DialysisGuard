"""
Alert Service — Alert creation, escalation, and management.
"""
import os
import sys
from datetime import datetime
from bson import ObjectId
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database


class AlertService:
    """Manages alerts with smart escalation."""
    
    # Escalation timings (seconds before auto-escalation)
    ESCALATION_TIMINGS = {
        "MODERATE": 300,   # 5 min → escalate to HIGH
        "HIGH": 180,       # 3 min → escalate to CRITICAL
        "CRITICAL": 60,    # 1 min → escalate to ALL
    }
    
    # Risk thresholds for alert generation
    ALERT_THRESHOLDS = {
        "MODERATE": 0.30,
        "HIGH": 0.50,
        "CRITICAL": 0.75,
    }
    
    def should_alert(self, risk_prob: float, recent_alerts: list = None) -> Optional[str]:
        """
        Determine if an alert should be generated.
        Returns severity level or None.
        """
        # Don't spam alerts — check if similar recent alert exists
        if recent_alerts:
            last_alert_time = recent_alerts[-1].get("created_at", "")
            # Simple cooldown: don't alert again within 30 seconds for same level
        
        if risk_prob >= self.ALERT_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif risk_prob >= self.ALERT_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif risk_prob >= self.ALERT_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        
        return None
    
    def create_alert(self, session_id: str, patient_id: str,
                      severity: str, risk_prob: float,
                      confidence: dict = None, 
                      nl_explanation: str = "",
                      top_features: list = None,
                      recommendations: list = None) -> dict:
        """Create and store an alert."""
        db = get_database()
        
        alert_doc = {
            "session_id": session_id,
            "patient_id": patient_id,
            "severity": severity,
            "risk_probability": round(risk_prob, 4),
            "confidence_lower": confidence.get("ci_lower") if confidence else None,
            "confidence_upper": confidence.get("ci_upper") if confidence else None,
            "message": self._generate_alert_message(severity, risk_prob),
            "nl_explanation": nl_explanation,
            "top_features": top_features or [],
            "recommendations": recommendations or [],
            "acknowledged": False,
            "escalation_level": 0,
            "created_at": datetime.utcnow().isoformat(),
            "acknowledged_at": None
        }
        
        result = db.alerts.insert_one(alert_doc)
        alert_doc["_id"] = result.inserted_id
        alert_doc["id"] = str(result.inserted_id)
        
        return alert_doc
    
    def acknowledge_alert(self, alert_id: str) -> dict:
        """Acknowledge an alert."""
        db = get_database()
        
        result = db.alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {
                "acknowledged": True,
                "acknowledged_at": datetime.utcnow().isoformat()
            }}
        )
        
        if result.matched_count == 0:
            return None
        
        return db.alerts.find_one({"_id": ObjectId(alert_id)})
    
    def check_escalation(self, session_id: str) -> List[dict]:
        """
        Check for alerts that need escalation.
        Returns list of escalated alerts.
        """
        db = get_database()
        
        unacked = list(db.alerts.find({
            "session_id": session_id,
            "acknowledged": False
        }))
        
        escalated = []
        now = datetime.utcnow()
        
        for alert in unacked:
            severity = alert["severity"]
            created = datetime.fromisoformat(alert["created_at"])
            elapsed = (now - created).total_seconds()
            
            timing = self.ESCALATION_TIMINGS.get(severity, 999999)
            
            if elapsed > timing and alert["escalation_level"] < 3:
                # Escalate
                new_level = alert["escalation_level"] + 1
                new_severity = self._escalate_severity(severity)
                
                db.alerts.update_one(
                    {"_id": alert["_id"]},
                    {"$set": {
                        "escalation_level": new_level,
                        "severity": new_severity,
                        "message": f"⬆️ ESCALATED: {self._generate_alert_message(new_severity, alert['risk_probability'])}"
                    }}
                )
                
                alert["escalation_level"] = new_level
                alert["severity"] = new_severity
                escalated.append(alert)
        
        return escalated
    
    def get_session_alerts(self, session_id: str) -> list:
        """Get all alerts for a session."""
        db = get_database()
        alerts = list(db.alerts.find(
            {"session_id": session_id}
        ).sort("created_at", -1))
        
        for a in alerts:
            a["id"] = str(a["_id"])
        
        return alerts
    
    def get_alert_stats(self) -> dict:
        """Get overall alert statistics."""
        db = get_database()
        
        total = db.alerts.count_documents({})
        unacked = db.alerts.count_documents({"acknowledged": False})
        
        by_severity = {}
        for sev in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
            by_severity[sev] = db.alerts.count_documents({"severity": sev})
        
        return {
            "total": total,
            "unacknowledged": unacked,
            "by_severity": by_severity
        }
    
    def _generate_alert_message(self, severity: str, risk_prob: float) -> str:
        """Generate alert message text."""
        messages = {
            "MODERATE": f"Moderate risk detected ({risk_prob:.0%}). Increased monitoring recommended.",
            "HIGH": f"High risk detected ({risk_prob:.0%}). Immediate assessment recommended.",
            "CRITICAL": f"⚠️ CRITICAL risk ({risk_prob:.0%}). Immediate intervention required."
        }
        return messages.get(severity, f"Risk level: {risk_prob:.0%}")
    
    def _escalate_severity(self, current: str) -> str:
        """Get next escalation level."""
        escalation = {
            "MODERATE": "HIGH",
            "HIGH": "CRITICAL",
            "CRITICAL": "CRITICAL"
        }
        return escalation.get(current, current)


# Singleton
alert_service = AlertService()
