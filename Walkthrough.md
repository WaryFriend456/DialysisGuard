# DialysisGuard — Implementation Walkthrough

## What Was Built

A complete AI-driven hemodialysis monitoring system with Explainable AI, built from scratch.

---

### ML Model (Trained & Saved)

| Metric | Value |
|--------|-------|
| Accuracy | 92% |
| AUC-ROC | 98.2% |
| Architecture | Attention-Augmented GRU (3 layers + Bahdanau Attention) |
| Training Data | 5000 patients × 30 time steps from [synthetic_hemodialysis_timeseries.csv](file:///G:/DialysisGuard/synthetic_hemodialysis_timeseries.csv) |

**Files:** [attention_gru.py](file:///G:/DialysisGuard/backend/ml/attention_gru.py), [train_model.py](file:///G:/DialysisGuard/backend/ml/train_model.py)  
**Artifacts:** [gru_model.h5](file:///G:/DialysisGuard/backend/ml/gru_model.h5), [scaler.pkl](file:///G:/DialysisGuard/backend/ml/scaler.pkl), [label_encoders.pkl](file:///G:/DialysisGuard/backend/ml/label_encoders.pkl), [feature_config.json](file:///G:/DialysisGuard/backend/ml/feature_config.json), [model_card.json](file:///G:/DialysisGuard/backend/ml/model_card.json)

---

### Backend — FastAPI (14 files)

**Entry:** [main.py](file:///G:/DialysisGuard/backend/main.py) → `uvicorn main:app --reload --port 8000`

| Component | File | Purpose |
|-----------|------|---------|
| Config | [config.py](file:///G:/DialysisGuard/backend/config.py) | Pydantic Settings, all paths |
| Database | [database.py](file:///G:/DialysisGuard/backend/database.py) | MongoDB connection |
| Schemas | [schemas.py](file:///G:/DialysisGuard/backend/models/schemas.py) | Pydantic models for all entities |
| Auth | [auth.py](file:///G:/DialysisGuard/backend/routes/auth.py) | JWT register/login/me |
| Patients | [patients.py](file:///G:/DialysisGuard/backend/routes/patients.py) | CRUD with search |
| Sessions | [sessions.py](file:///G:/DialysisGuard/backend/routes/sessions.py) | Start/stop/report |
| Predictions | [predictions.py](file:///G:/DialysisGuard/backend/routes/predictions.py) | Single + full risk assessment |
| Alerts | [alerts.py](file:///G:/DialysisGuard/backend/routes/alerts.py) | List/ack/stats |
| XAI Routes | [explanations.py](file:///G:/DialysisGuard/backend/routes/explanations.py) | SHAP/attention/what-if/counterfactual/sensitivity/model-card |
| ML Service | [ml_service.py](file:///G:/DialysisGuard/backend/services/ml_service.py) | Model loading, MC Dropout, predictions |
| Simulation | [simulation_service.py](file:///G:/DialysisGuard/backend/services/simulation_service.py) | Physiological vital generation + anomaly detection |
| XAI Service | [xai_service.py](file:///G:/DialysisGuard/backend/services/xai_service.py) | SHAP, attention, what-if, counterfactual, NL, sensitivity |
| Alert Service | [alert_service.py](file:///G:/DialysisGuard/backend/services/alert_service.py) | Smart escalation |
| WebSocket | [realtime.py](file:///G:/DialysisGuard/backend/websocket/realtime.py) | Real-time streaming |

---

### Frontend — Next.js (12 pages/components)

**Run:** `npm run dev` (from `frontend/`)

| Page | Purpose |
|------|---------|
| [Login](file:///G:/DialysisGuard/frontend/src/app/login/page.js) | Glassmorphism login with gradient orbs |
| [Register](file:///G:/DialysisGuard/frontend/src/app/register/page.js) | Role-based registration |
| [Doctor Dashboard](file:///G:/DialysisGuard/frontend/src/app/dashboard/doctor/page.js) | Stats grid, recent patients/alerts, quick actions |
| [Caregiver Dashboard](file:///G:/DialysisGuard/frontend/src/app/dashboard/caregiver/page.js) | Simplified view with alerts |
| [Command Center](file:///G:/DialysisGuard/frontend/src/app/dashboard/command/page.js) | Multi-patient grid sorted by severity |
| [Patients](file:///G:/DialysisGuard/frontend/src/app/patients/page.js) | CRUD with add modal (all clinical fields) |
| [Monitor](file:///G:/DialysisGuard/frontend/src/app/monitor/page.js) | **Centerpiece** — Real-time WebSocket monitoring |
| [Alerts](file:///G:/DialysisGuard/frontend/src/app/alerts/page.js) | Filter/acknowledge alerts |
| [Model Info](file:///G:/DialysisGuard/frontend/src/app/model-info/page.js) | Full model transparency card |

---

### Key Innovative Features

1. **Risk Gauge** — Color-coded percentage with confidence interval and trend arrow
2. **SHAP Waterfall** — Real-time feature attribution bars (positive/negative)
3. **Temporal Attention** — Bar chart showing which time steps the model focused on
4. **NL Explanation** — Human-readable paragraph explaining the risk assessment
5. **Risk Forecast** — 5-step prediction cone
6. **Anomaly Markers** — Z-score + rate-of-change detection, flashing badges
7. **Screen Flash + Audio** — Critical alert visual/audio indicators
8. **Smart Escalation** — Auto-escalates unacknowledged alerts
9. **Session Report** — Auto-generated on session end
10. **BP/HR Sparklines** — Mini bar charts showing recent vitals trajectory

---

### Verification Results

| Check | Result |
|-------|--------|
| FastAPI import test | ✅ Pass |
| Next.js production build | ✅ Pass |
| Model training | ✅ 92% accuracy, 98.2% AUC |
| All artifacts saved | ✅ 5 files in `backend/ml/` |

---

### How to Run

```bash
# Backend (requires MongoDB running)
cd G:\DialysisGuard\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd G:\DialysisGuard\frontend
npm run dev
```

Open `http://localhost:3000` → Register → Add Patient → Start Monitor 🔴
