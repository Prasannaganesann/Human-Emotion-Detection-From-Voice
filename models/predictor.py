"""
predictor.py
============
Inference engine wrapping the trained model, scaler, and label encoder.
Provides a clean API for both the Streamlit app and FastAPI backend.

Changes in v2:
- Added inference_time_ms to every result
- Added top_k_predictions (sorted list) to every result
- Added model_version field
- Improved uncertainty handling: still shows real probabilities even
  when uncertain, instead of hiding them
- Lower UNCERTAINTY_THRESHOLD to 0.30 (matches better-calibrated demo model)
"""

import json
import logging
import time
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    BEST_MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE,
    EMOTION_EMOJIS, EMOTION_COLORS, MODELS_DIR
)
from utils.preprocessing import preprocess_audio, get_audio_info
from utils.feature_extraction import extract_features

logger = logging.getLogger(__name__)

# Confidence threshold — predictions below this are flagged as uncertain.
# Set to 0.30 (was 0.35) to work correctly with the improved demo model.
UNCERTAINTY_THRESHOLD = 0.30

# Model version tag injected into every result dict
MODEL_VERSION = "v2.0"


class EmotionPredictor:
    """
    Loads a model + scaler + label encoder and exposes ``predict()``.

    Designed for Streamlit's ``st.cache_resource`` — one instance per
    unique model path is cached, enabling hot-swap without reloading.

    Usage
    -----
    predictor = EmotionPredictor()
    predictor.load("saved_models/svm_rbf_model.pkl")
    result = predictor.predict_from_bytes(audio_bytes)
    """

    def __init__(self):
        self._loaded   = False
        self.model     = None
        self.scaler    = None
        self.le        = None
        self.classes   = []
        self.model_name = ""

    # ─────────────────────────────────────────
    #  Loading
    # ─────────────────────────────────────────

    def load(self, model_path: str = None) -> None:
        """Load model, scaler, and label encoder from disk."""
        model_file = Path(model_path) if model_path else BEST_MODEL_FILE

        for fpath, label in [
            (model_file,        "Model"),
            (SCALER_FILE,       "Scaler"),
            (LABEL_ENCODER_FILE,"Label encoder"),
        ]:
            if not fpath.exists():
                raise FileNotFoundError(
                    f"{label} not found at {fpath}. "
                    "Run `python generate_demo_model.py` first."
                )

        self.model      = joblib.load(model_file)
        self.scaler     = joblib.load(SCALER_FILE)
        self.le         = joblib.load(LABEL_ENCODER_FILE)
        self.classes    = self.le.classes_.tolist()
        self.model_name = model_file.stem
        self._loaded    = True
        logger.info(f"Loaded: {self.model_name}  |  classes={self.classes}")

    # ─────────────────────────────────────────
    #  Core Prediction
    # ─────────────────────────────────────────

    def predict(self, y: np.ndarray, sr: int,
                audio_info: dict = None) -> dict:
        """
        Run inference on a preprocessed audio array.

        Returns a rich result dict with:
          - emotion, confidence, probabilities
          - top_k_predictions (sorted list)
          - is_uncertain flag + human-readable reason
          - inference_time_ms
          - model_version
          - emoji + color for UI
        """
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        feat_vec   = extract_features(y, sr)
        feat_scaled = self.scaler.transform(feat_vec.reshape(1, -1))

        pred_idx  = self.model.predict(feat_scaled)[0]
        proba_arr = self.model.predict_proba(feat_scaled)[0]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        emotion    = self.le.inverse_transform([pred_idx])[0]
        confidence = float(proba_arr[pred_idx])
        probs      = {cls: round(float(p), 4)
                      for cls, p in zip(self.classes, proba_arr)}

        # Sorted top-k predictions (always show top 3 in UI)
        top_k = sorted(
            [{"emotion": e, "score": round(float(p), 4),
              "pct": f"{p*100:.1f}%",
              "emoji": EMOTION_EMOJIS.get(e, "🎙️"),
              "color": EMOTION_COLORS.get(e, "#94A3B8")}
             for e, p in probs.items()],
            key=lambda x: x["score"], reverse=True
        )

        is_uncertain = confidence < UNCERTAINTY_THRESHOLD

        # Provide a human-readable uncertainty reason
        if is_uncertain:
            top2_margin = top_k[0]["score"] - top_k[1]["score"]
            if top2_margin < 0.08:
                uncertainty_reason = "Multiple emotions equally likely"
            else:
                uncertainty_reason = "Low overall confidence"
        else:
            uncertainty_reason = None

        return {
            # Core prediction
            "emotion":            emotion,
            "confidence":         round(confidence, 4),
            "confidence_pct":     f"{confidence * 100:.1f}%",
            "is_uncertain":       is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            # Per-class probabilities
            "probabilities":      probs,
            "top_k":              top_k,
            # UI helpers
            "emoji":  EMOTION_EMOJIS.get(emotion, "🎙️"),
            "color":  EMOTION_COLORS.get(emotion, "#94A3B8"),
            # Metadata
            "model_name":         self.model_name,
            "model_version":      MODEL_VERSION,
            "inference_time_ms":  elapsed_ms,
            "audio_info":         audio_info or {},
        }

    # ─────────────────────────────────────────
    #  Convenience Wrappers
    # ─────────────────────────────────────────

    def predict_from_file(self, file_path: str) -> dict:
        """Preprocess + predict from a file path."""
        y, sr = preprocess_audio(file_path=file_path)
        info  = get_audio_info(y, sr)
        return self.predict(y, sr, audio_info=info)

    def predict_from_bytes(self, audio_bytes: bytes,
                           filename: str = "upload") -> dict:
        """Preprocess + predict from raw bytes."""
        y, sr = preprocess_audio(audio_bytes=audio_bytes)
        info  = get_audio_info(y, sr)
        info["filename"] = filename
        return self.predict(y, sr, audio_info=info)

    def predict_from_array(self, y: np.ndarray, sr: int) -> dict:
        """Preprocess + predict from a NumPy array (e.g. microphone)."""
        from utils.preprocessing import trim_silence, normalize_audio, pad_or_trim
        y = trim_silence(y)
        y = normalize_audio(y)
        y = pad_or_trim(y, sr)
        info = get_audio_info(y, sr)
        return self.predict(y, sr, audio_info=info)

    def predict_batch(self, file_paths: list[str]) -> list[dict]:
        """Predict emotions for a list of file paths."""
        results = []
        for fp in file_paths:
            try:
                results.append(self.predict_from_file(fp))
            except Exception as e:
                results.append({"file": fp, "error": str(e)})
        return results

    def predict_batch_bytes(self,
                            items: list[tuple[bytes, str]]) -> list[dict]:
        """Predict emotions for a batch of (audio_bytes, filename) tuples."""
        results = []
        for audio_bytes, filename in items:
            try:
                results.append(self.predict_from_bytes(audio_bytes, filename))
            except Exception as e:
                results.append({"file": filename, "error": str(e)})
        return results

    # ─────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────

    @staticmethod
    def list_available_models() -> list[str]:
        """Return all saved model pkl files."""
        return [str(p) for p in MODELS_DIR.glob("*_model.pkl")]
