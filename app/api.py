"""
api.py  –  FastAPI REST Backend
================================
Bonus API endpoint for integrating emotion detection into any front-end
or external service.

Endpoints:
  POST /predict          – Upload a .wav file, get emotion prediction
  GET  /history          – Last N prediction records
  GET  /stats            – Aggregate emotion statistics
  GET  /models           – List available saved models
  GET  /health           – Health check

Run:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.predictor import EmotionPredictor
from utils.database import (
    init_db, save_prediction, get_predictions,
    get_emotion_stats, get_trend_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="VoiceEmo API",
    description="Human Emotion Detection from Voice – REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to specific origins in production
    allow_credentials=False,      # FIX: cannot use True with wildcard origins (browser-rejected)
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialise DB on startup
init_db()

# Load predictor once
_predictor: Optional[EmotionPredictor] = None


def get_predictor() -> EmotionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor()
        _predictor.load()
    return _predictor


# ─────────────────────────────────────────────
#  Response Schemas
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Response schema for /predict. Matches EmotionPredictor.predict() result dict."""
    emotion:            str
    confidence:         float
    confidence_pct:     str
    is_uncertain:       bool
    uncertainty_reason: str | None
    emoji:              str
    color:              str
    probabilities:      dict
    top_k:              list
    model_name:         str
    model_version:      str
    inference_time_ms:  float
    audio_info:         dict


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "VoiceEmo API v1.0"}


@app.get("/models", tags=["System"])
def list_models():
    """List all available trained model files."""
    from config import MODELS_DIR
    files = [p.name for p in MODELS_DIR.glob("*_model.pkl")]
    return {"available_models": files, "count": len(files)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, etc.)"),
    session_id: Optional[str] = Query(None, description="Optional session ID"),
):
    """
    Upload an audio file and receive an emotion prediction.

    Returns the predicted emotion, confidence score,
    full probability distribution, and audio metadata.
    """
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use: {allowed}")

    try:
        audio_bytes = await file.read()
        predictor   = get_predictor()
        result      = predictor.predict_from_bytes(audio_bytes, filename=file.filename)

        # Persist to DB
        save_prediction(
            emotion=result["emotion"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_name=result["model_name"],
            filename=file.filename,
            duration_s=result["audio_info"].get("duration_s"),
            session_id=session_id,
        )

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not ready: {e}. Train first.")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(500, f"Prediction failed: {e}")


@app.get("/history", tags=["Data"])
def history(
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    session_id: Optional[str] = Query(None),
):
    """Retrieve recent prediction history."""
    records = get_predictions(limit=limit, session_id=session_id)
    return {"records": records, "count": len(records)}


@app.get("/stats", tags=["Data"])
def stats():
    """Aggregate emotion statistics across all predictions."""
    return {"stats": get_emotion_stats()}


@app.get("/trend", tags=["Data"])
def trend(n: int = Query(50, ge=5, le=500)):
    """Return the last N predictions for trend analysis."""
    return {"trend": get_trend_data(n=n)}


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple audio files"),
    session_id: Optional[str] = Query(None),
):
    """
    Upload multiple audio files and receive predictions for each.
    """
    predictor = get_predictor()
    results = []
    for file in files:
        try:
            audio_bytes = await file.read()
            result = predictor.predict_from_bytes(audio_bytes, filename=file.filename)
            save_prediction(
                emotion=result["emotion"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                model_name=result["model_name"],
                filename=file.filename,
                duration_s=result["audio_info"].get("duration_s"),
                session_id=session_id,
            )
            results.append(result)
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
    return {"results": results, "count": len(results)}

