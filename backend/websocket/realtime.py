"""
WebSocket Real-Time Monitoring — Streams vitals, predictions, XAI, anomalies.

Each connected client receives data every ~2.5 seconds simulating a dialysis
session, with the physiological simulation engine generating vital signs and
the GRU model predicting instability in real-time.
"""
import asyncio
import traceback
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from bson import ObjectId
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from config import settings
from services.ml_service import ml_service
from services.xai_service import xai_service
from services.simulation_service import simulator
from services.alert_service import alert_service


class MonitoringManager:
    """Manages active WebSocket monitoring sessions."""

    def __init__(self):
        self.active_connections: dict = {}  # session_id -> WebSocket

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    def is_connected(self, session_id: str) -> bool:
        return session_id in self.active_connections


manager = MonitoringManager()


async def _safe_send(websocket: WebSocket, data: dict) -> bool:
    """Send JSON, return False if socket is already closed."""
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
    except Exception:
        pass
    return False


async def websocket_monitor(websocket: WebSocket, session_id: str):
    """
    Main WebSocket handler for real-time dialysis monitoring.

    Flow:
    1. Accept connection, load session + patient data
    2. Start simulation loop (30 time steps, ~2.5s interval)
    3. Each step: generate vitals -> predict -> XAI -> anomaly detect -> stream
    4. On disconnect or completion: save session data
    """
    await manager.connect(websocket, session_id)
    db = get_database()

    try:
        # Load session
        session = db.sessions.find_one({"_id": ObjectId(session_id)})
        if not session:
            await _safe_send(websocket, {"error": "Session not found"})
            return

        # Load patient
        patient = db.patients.find_one({"_id": ObjectId(session["patient_id"])})
        if not patient:
            await _safe_send(websocket, {"error": "Patient not found"})
            return

        patient_data = {k: v for k, v in patient.items() if k != "_id"}
        patient_data["id"] = str(patient["_id"])

        # Determine risk profile
        risk_profile = session.get("risk_profile") or simulator.pick_risk_profile()

        if not await _safe_send(websocket, {
            "type": "session_start",
            "session_id": session_id,
            "patient_id": str(patient["_id"]),
            "patient_name": patient.get("name", f"Patient {str(patient['_id'])[:6]}"),
            "risk_profile": risk_profile,
            "total_steps": settings.SIMULATION_TIME_STEPS,
            "interval_seconds": settings.SIMULATION_INTERVAL_SECONDS
        }):
            return  # Client already gone

        # Simulation loop
        previous_steps = []
        predictions_history = []
        session_alerts = []

        for step in range(settings.SIMULATION_TIME_STEPS):
            if not manager.is_connected(session_id):
                break

            try:
                # 1. Generate vital signs
                step_data = simulator.generate_step(
                    patient_data, previous_steps, risk_profile
                )
                previous_steps.append(step_data)

                # 2. Preprocess and predict
                X = ml_service.preprocess_sequence(previous_steps)
                uncertainty = ml_service.predict_with_uncertainty(X)
                risk_prob = float(uncertainty['mean'])
                risk_level = ml_service.get_risk_level(risk_prob)
                predictions_history.append(risk_prob)

                # 3. XAI computations
                # Feature importance — only every 3rd step to reduce latency
                top_features = []
                if step % 3 == 0:
                    try:
                        shap_data = xai_service._approximate_feature_importance(X)
                        top_features = shap_data.get('top_contributors', [])[:5]
                    except Exception as xe:
                        print(f"  XAI error step {step}: {xe}")

                # Attention weights
                attention_data = {'attention_weights': [], 'peak_attention_time': ''}
                try:
                    attention_data = xai_service.get_attention_weights(X)
                except Exception:
                    pass

                # Risk trend
                trend = xai_service.calculate_risk_trend(predictions_history)

                # Risk forecast
                forecast = xai_service.forecast_risk(risk_prob, trend)

                # NL explanation
                nl_explanation = ""
                try:
                    nl_explanation = xai_service.generate_nl_explanation(
                        risk_prob, risk_level, top_features,
                        attention_data, uncertainty, [], trend
                    )
                except Exception:
                    nl_explanation = f"Risk is {risk_level.upper()} ({risk_prob:.0%})"

                # 4. Anomaly detection
                anomalies = []
                try:
                    anomalies = simulator.detect_anomalies(step_data, previous_steps[:-1])
                except Exception:
                    pass

                # 5. Check if alert needed
                alert = None
                severity = alert_service.should_alert(risk_prob, session_alerts)
                if severity:
                    try:
                        alert_doc = alert_service.create_alert(
                            session_id=session_id,
                            patient_id=str(patient["_id"]),
                            severity=severity,
                            risk_prob=risk_prob,
                            confidence=uncertainty,
                            nl_explanation=nl_explanation,
                            top_features=top_features,
                            recommendations=ml_service.get_recommendations(risk_level, top_features)
                        )
                        session_alerts.append(alert_doc)
                        alert = {
                            "id": str(alert_doc.get("_id", "")),
                            "severity": severity,
                            "message": alert_doc["message"],
                            "escalation_level": 0
                        }
                    except Exception as ae:
                        print(f"  Alert error step {step}: {ae}")

                # 6. Check for escalations
                escalation_alerts = []
                try:
                    escalated = alert_service.check_escalation(session_id)
                    escalation_alerts = [{
                        "id": str(e.get("_id", "")),
                        "severity": e["severity"],
                        "escalation_level": e.get("escalation_level", 0),
                        "message": f"⬆️ ESCALATED: {e.get('message', '')}"
                    } for e in escalated]
                except Exception:
                    pass

                # 7. Build and send payload
                payload = {
                    "type": "monitoring_data",
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
                            "mean": round(float(uncertainty['mean']), 4),
                            "lower": round(float(uncertainty['ci_lower']), 4),
                            "upper": round(float(uncertainty['ci_upper']), 4),
                            "std": round(float(uncertainty['std']), 4)
                        }
                    },
                    "xai": {
                        "top_features": top_features,
                        "attention_weights": attention_data.get('attention_weights', []),
                        "nl_explanation": nl_explanation,
                        "risk_trend": trend,
                        "risk_forecast_5step": forecast
                    },
                    "anomalies": anomalies,
                    "alert": alert,
                    "escalation_alerts": escalation_alerts
                }

                if not await _safe_send(websocket, payload):
                    break  # Client disconnected mid-send

                # Store in session (fire-and-forget — don't let DB errors kill the loop)
                try:
                    db.sessions.update_one(
                        {"_id": ObjectId(session_id)},
                        {"$push": {
                            "time_series_data": {
                                "step": step,
                                "bp": step_data["Current_BP"],
                                "hr": step_data["Current_HR"],
                                "time_minutes": step_data["Time_Minutes"]
                            },
                            "predictions": {
                                "step": step,
                                "risk_probability": round(risk_prob, 4),
                                "risk_level": risk_level
                            }
                        }}
                    )
                except Exception:
                    pass

                # Wait before next step
                await asyncio.sleep(settings.SIMULATION_INTERVAL_SECONDS)

            except Exception as e:
                print(f"Step {step} error: {e}")
                traceback.print_exc()
                # DON'T try to send error to client — socket may be closed
                continue

        # Session complete
        await _safe_send(websocket, {
            "type": "session_complete",
            "total_steps": len(previous_steps),
            "final_risk": predictions_history[-1] if predictions_history else 0,
            "total_alerts": len(session_alerts)
        })

    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
    finally:
        manager.disconnect(session_id)

        # Update session status
        try:
            db.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"status": "completed", "end_time": datetime.utcnow().isoformat()}}
            )
        except Exception:
            pass
