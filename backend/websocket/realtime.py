"""
WebSocket realtime monitoring for dialysis sessions.
"""

import asyncio
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime

from bson import ObjectId
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from database import get_database
from models.schemas import SessionStatus
from services.alert_service import alert_service
from services.ml_service import ml_service
from services.simulation_service import simulator
from services.xai_service import xai_service
from routes.auth import decode_token


logger = logging.getLogger("dialysisguard.realtime")


class MonitoringManager:
    """Tracks active websocket connections."""

    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    def is_connected(self, session_id: str) -> bool:
        return session_id in self.active_connections


manager = MonitoringManager()


async def _safe_send(websocket: WebSocket, data: dict) -> bool:
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
    except Exception:
        return False
    return False


def _normalize_saved_steps(saved_steps: list) -> list:
    normalized = []
    for entry in saved_steps or []:
        if "Current_BP" in entry and "Current_HR" in entry:
            normalized.append(entry)
    return normalized


async def websocket_monitor(websocket: WebSocket, session_id: str):
    """Stream monitoring updates for a resumable session."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.accept()
        await _safe_send(websocket, {"type": "error", "message": "Unauthorized"})
        await websocket.close()
        return

    try:
        user = decode_token(token)
    except HTTPException:
        await websocket.accept()
        await _safe_send(websocket, {"type": "error", "message": "Unauthorized"})
        await websocket.close()
        return

    if user.get("role") == "super_admin" or not user.get("org_id") or user.get("must_change_password"):
        await websocket.accept()
        await _safe_send(websocket, {"type": "error", "message": "Unauthorized"})
        await websocket.close()
        return

    await manager.connect(websocket, session_id)
    db = get_database()
    completed_normally = False
    org_id = user["org_id"]

    try:
        try:
            session = db.sessions.find_one({"_id": ObjectId(session_id), "org_id": org_id})
        except Exception:
            session = None

        if not session:
            await _safe_send(websocket, {"type": "error", "message": "Session not found"})
            return

        if session.get("status") in {SessionStatus.COMPLETED.value, SessionStatus.STOPPED.value}:
            await _safe_send(websocket, {"type": "error", "message": "Session is no longer resumable"})
            return

        patient = db.patients.find_one({"_id": ObjectId(session["patient_id"]), "org_id": org_id})
        if not patient:
            await _safe_send(websocket, {"type": "error", "message": "Patient not found"})
            return

        patient_data = {k: v for k, v in patient.items() if k != "_id"}
        patient_data["id"] = str(patient["_id"])

        previous_steps = _normalize_saved_steps(session.get("time_series_data", []))
        start_index = int(session.get("current_step", len(previous_steps)))
        previous_steps = previous_steps[:start_index]
        risk_profile = session.get("risk_profile") or simulator.pick_risk_profile()
        total_steps = int(session.get("total_steps", settings.SIMULATION_TIME_STEPS))
        predictions_history = [
            float(pred.get("risk_probability", 0.0))
            for pred in session.get("predictions", [])
        ]
        session_alerts = alert_service.get_session_alerts(session_id, org_id)

        db.sessions.update_one(
            {"_id": ObjectId(session_id), "org_id": org_id},
            {
                "$set": {
                    "status": SessionStatus.ACTIVE.value,
                    "risk_profile": risk_profile,
                    "total_steps": total_steps,
                    "current_step": start_index,
                }
            },
        )

        if not await _safe_send(
            websocket,
            {
                "type": "session_start",
                "session_id": session_id,
                "patient_id": str(patient["_id"]),
                "patient_name": patient.get("name", f"Patient {str(patient['_id'])[:6]}"),
                "risk_profile": risk_profile,
                "total_steps": total_steps,
                "current_step": start_index,
                "interval_seconds": settings.simulation_interval_seconds,
                "interval_seconds_min": settings.SIMULATION_INTERVAL_MIN_SECONDS,
                "interval_seconds_max": settings.SIMULATION_INTERVAL_MAX_SECONDS,
                "is_resume": start_index > 0,
            },
        ):
            return

        for step in range(start_index, total_steps):
            if not manager.is_connected(session_id):
                break

            step_started = time.monotonic()

            try:
                step_data = simulator.generate_step(patient_data, previous_steps, risk_profile)
                stored_step = {"step": step, **step_data}
                previous_steps.append(step_data)

                X = ml_service.preprocess_sequence(previous_steps)
                deterministic = ml_service.predict(X)
                uncertainty = ml_service.predict_with_uncertainty(
                    X, n_passes=settings.REALTIME_MC_DROPOUT_PASSES
                )
                risk_prob = float(deterministic["probability"])
                risk_level = deterministic["risk_level"]
                predictions_history.append(risk_prob)

                top_features = xai_service.approximate_feature_importance_fast(X)[
                    "top_contributors"
                ][:5]
                attention_data = xai_service.get_attention_weights(X)
                trend = xai_service.calculate_risk_trend(predictions_history)
                forecast = xai_service.forecast_risk(risk_prob, trend)
                anomalies = simulator.detect_anomalies(step_data, previous_steps[:-1])
                nl_explanation = xai_service.generate_nl_explanation(
                    risk_prob,
                    risk_level,
                    top_features,
                    attention_data,
                    uncertainty,
                    anomalies,
                    trend,
                )

                alert = None
                severity = alert_service.should_alert(risk_prob, session_alerts)
                if severity:
                    alert_doc = alert_service.create_alert(
                        session_id=session_id,
                        patient_id=str(patient["_id"]),
                        org_id=org_id,
                        severity=severity,
                        risk_prob=risk_prob,
                        confidence=uncertainty,
                        nl_explanation=nl_explanation,
                        top_features=top_features,
                        recommendations=ml_service.get_recommendations(risk_level, top_features),
                    )
                    session_alerts.append(alert_doc)
                    alert = {
                        "id": str(alert_doc.get("_id", "")),
                        "severity": severity,
                        "message": alert_doc["message"],
                        "escalation_level": int(alert_doc.get("escalation_level", 0)),
                    }

                escalated = alert_service.check_escalation(session_id, org_id)
                escalation_alerts = [
                    {
                        "id": str(item.get("_id", "")),
                        "severity": item["severity"],
                        "escalation_level": item.get("escalation_level", 0),
                        "message": item.get("message", ""),
                    }
                    for item in escalated
                ]

                payload = {
                    "type": "monitoring_data",
                    "session_id": session_id,
                    "patient_id": str(patient["_id"]),
                    "patient_name": patient.get("name"),
                    "step": step,
                    "time_minutes": step_data["Time_Minutes"],
                    "vitals": {
                        "bp": step_data["Current_BP"],
                        "hr": step_data["Current_HR"],
                        "bp_change": step_data.get("BP_Change", 0),
                        "hr_change": step_data.get("HR_Change", 0),
                    },
                    "prediction": {
                        "risk_probability": round(risk_prob, 4),
                        "risk_level": risk_level,
                        "confidence": {
                            "mean": round(float(uncertainty["mean"]), 4),
                            "lower": round(float(uncertainty["ci_lower"]), 4),
                            "upper": round(float(uncertainty["ci_upper"]), 4),
                            "std": round(float(uncertainty["std"]), 4),
                        },
                    },
                    "xai": {
                        "top_features": top_features,
                        "attention_weights": attention_data.get("attention_weights", []),
                        "nl_explanation": nl_explanation,
                        "risk_trend": trend,
                        "risk_forecast_5step": forecast,
                    },
                    "anomalies": anomalies,
                    "alert": alert,
                    "escalation_alerts": escalation_alerts,
                }

                db.sessions.update_one(
                    {"_id": ObjectId(session_id), "org_id": org_id},
                    {
                        "$set": {
                            "current_step": step + 1,
                            "status": SessionStatus.ACTIVE.value,
                            "risk_profile": risk_profile,
                            "total_steps": total_steps,
                        },
                        "$push": {
                            "time_series_data": stored_step,
                            "predictions": {
                                "step": step,
                                "risk_probability": round(risk_prob, 4),
                                "risk_level": risk_level,
                            },
                            "explanations": {
                                "step": step,
                                "top_features": top_features,
                                "attention_weights": attention_data.get("attention_weights", []),
                                "risk_trend": trend,
                                "nl_explanation": nl_explanation,
                            },
                        },
                    },
                )

                if not await _safe_send(websocket, payload):
                    break

                elapsed = time.monotonic() - step_started
                interval_target = random.uniform(
                    settings.SIMULATION_INTERVAL_MIN_SECONDS,
                    settings.SIMULATION_INTERVAL_MAX_SECONDS,
                )
                sleep_for = max(0.0, interval_target - elapsed)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            except Exception as exc:
                logger.error("Monitoring step %s failed: %s", step, exc)
                traceback.print_exc()
                await _safe_send(
                    websocket,
                    {"type": "error", "message": f"Monitoring step failed: {exc}"},
                )
                break

        latest_session = db.sessions.find_one({"_id": ObjectId(session_id), "org_id": org_id})
        if latest_session and int(latest_session.get("current_step", 0)) >= total_steps:
            completed_normally = True
            db.sessions.update_one(
                {"_id": ObjectId(session_id), "org_id": org_id},
                {
                    "$set": {
                        "status": SessionStatus.COMPLETED.value,
                        "end_time": datetime.utcnow().isoformat(),
                    }
                },
            )
            await _safe_send(
                websocket,
                {
                    "type": "session_complete",
                    "session_id": session_id,
                    "total_steps": total_steps,
                    "final_risk": predictions_history[-1] if predictions_history else 0,
                    "total_alerts": len(session_alerts),
                },
            )

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)
    except Exception as exc:
        logger.error("Realtime websocket error for session %s: %s", session_id, exc)
        traceback.print_exc()
    finally:
        manager.disconnect(session_id)
        try:
            session = db.sessions.find_one({"_id": ObjectId(session_id), "org_id": org_id})
            if not session:
                return

            status = session.get("status")
            explicit_stop = bool(session.get("explicit_stop", False))
            current_step = int(session.get("current_step", 0))
            total_steps = int(session.get("total_steps", settings.SIMULATION_TIME_STEPS))

            if explicit_stop or status in {
                SessionStatus.COMPLETED.value,
                SessionStatus.STOPPED.value,
            }:
                return

            if completed_normally or current_step >= total_steps:
                db.sessions.update_one(
                    {"_id": ObjectId(session_id), "org_id": org_id},
                    {
                        "$set": {
                            "status": SessionStatus.COMPLETED.value,
                            "end_time": datetime.utcnow().isoformat(),
                        }
                    },
                )
            else:
                db.sessions.update_one(
                    {"_id": ObjectId(session_id), "org_id": org_id},
                    {
                        "$set": {
                            "status": SessionStatus.PAUSED.value,
                            "end_time": None,
                        }
                    },
                )
        except Exception as exc:
            logger.warning("Failed to finalize session %s state: %s", session_id, exc)
