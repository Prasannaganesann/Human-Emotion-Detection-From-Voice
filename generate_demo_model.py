"""
generate_demo_model.py
======================
Quick-start helper: generates a SYNTHETIC demo model so you can run the
Streamlit app IMMEDIATELY without downloading RAVDESS first.

The demo model is trained on random audio-like features and will produce
plausible-looking (but not accurate) predictions.
Replace it with the real trained model after running the full pipeline.

Usage:
    python generate_demo_model.py
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

from config import (
    MODELS_DIR, BEST_MODEL_FILE, SCALER_FILE,
    LABEL_ENCODER_FILE, FEATURE_NAMES_FILE,
    TARGET_EMOTIONS, RANDOM_STATE
)
from utils.feature_extraction import get_feature_names, save_feature_names


def main():
    import joblib
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.datasets import make_classification

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature names ─────────────────────────────────────────────────
    feature_names = get_feature_names()
    save_feature_names(feature_names)
    n_features = len(feature_names)
    logger.info(f"Feature vector size: {n_features}")

    # ── Synthetic training data ───────────────────────────────────────
    n_classes = len(TARGET_EMOTIONS)
    X, y_raw = make_classification(
        n_samples=1200,
        n_features=n_features,
        n_informative=min(60, n_features // 3),
        n_redundant=20,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )

    # ── Label encoder ─────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(TARGET_EMOTIONS)
    # Map raw int → valid emotion label
    y = np.array([le.transform([TARGET_EMOTIONS[i % n_classes]])[0]
                  for i in y_raw])

    # ── Scaler ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── SVM model ─────────────────────────────────────────────────────
    clf = SVC(kernel="rbf", C=5, gamma="scale",
              probability=True, random_state=RANDOM_STATE)
    clf.fit(X_scaled, y)
    logger.info("Demo SVM trained.")

    # ── Save artefacts ────────────────────────────────────────────────
    joblib.dump(clf,    BEST_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le,     LABEL_ENCODER_FILE)

    # Also save as named model file
    joblib.dump(clf, MODELS_DIR / "svm_rbf_model.pkl")

    logger.info(f"✅  Demo model saved → {BEST_MODEL_FILE}")
    logger.info(f"✅  Scaler saved     → {SCALER_FILE}")
    logger.info(f"✅  Label encoder    → {LABEL_ENCODER_FILE}")
    logger.info("\n⚠️  This is a DEMO model trained on synthetic data.")
    logger.info("    Download RAVDESS and run training/train_model.py for real accuracy.")


if __name__ == "__main__":
    main()
