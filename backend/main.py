"""
DialysisGuard FastAPI Application — Main Entry Point

Run with: uvicorn main:app --reload --port 8000
"""
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from database import get_database, close_database
from services.ml_service import ml_service
from services.xai_service import xai_service

# Routes
from routes.auth import router as auth_router
from routes.patients import router as patients_router
from routes.sessions import router as sessions_router
from routes.predictions import router as predictions_router
from routes.alerts import router as alerts_router
from routes.explanations import router as explanations_router

# WebSocket
from websocket.realtime import websocket_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    print("🚀 Starting DialysisGuard API...")
    
    # Initialize database
    get_database()
    print("✅ MongoDB connected")
    
    # Initialize ML model
    ml_service.initialize()
    
    # Initialize XAI service
    xai_service.initialize(ml_service)
    
    print("🚀 DialysisGuard API is ready!")
    print(f"📄 API Docs: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    close_database()
    print("👋 DialysisGuard API shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Driven Real-Time Hemodialysis Monitoring with Explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(auth_router)
app.include_router(patients_router)
app.include_router(sessions_router)
app.include_router(predictions_router)
app.include_router(alerts_router)
app.include_router(explanations_router)


# WebSocket endpoint
@app.websocket("/ws/monitor/{session_id}")
async def ws_monitor(websocket: WebSocket, session_id: str):
    """Real-time monitoring WebSocket endpoint."""
    await websocket_monitor(websocket, session_id)


# Health check
@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "model_loaded": ml_service._initialized
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
