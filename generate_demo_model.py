"""
generate_demo_model.py
======================
Generates a DEMO model with well-separated synthetic clusters so the
Streamlit app produces varied, realistic-looking predictions instead
of always returning "Uncertain 33.2%".

Root cause of the "always uncertain" bug:
  - The old synthetic data had no real cluster separation (make_classification
    with high n_informative in a very high-dimensional space → near-uniform
    probabilities after Platt scaling → confidence ≈ 1/N ≈ 20-33%).
  - This version uses explicit per-class Gaussian clusters with large
    inter-class separation so the SVM learns confident decision boundaries.

Usage:
    python generate_demo_model.py
"""

import sys
import json
import logging
import time
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


def make_separated_dataset(
    n_per_class: int,
    n_features: int,
    n_classes: int,
    separation: float = 6.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build synthetic data where each class occupies a distinct region of
    feature space — guaranteeing the trained SVM will have confident
    decision boundaries and produce varied probability outputs at inference.

    Parameters
    ----------
    separation : float
        Distance between class centroids in std-dev units.
        Higher → more confident predictions.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    # Pick class centroids that are maximally spread in feature space
    centroids = rng.standard_normal((n_classes, n_features)) * separation

    X_parts, y_parts = [], []
    for cls_idx in range(n_classes):
        # Each class has its own covariance (not just spherical noise)
        scale = rng.uniform(0.6, 1.2, size=n_features)
        X_cls = centroids[cls_idx] + rng.normal(0, 1, size=(n_per_class, n_features)) * scale
        X_parts.append(X_cls)
        y_parts.append(np.full(n_per_class, cls_idx, dtype=int))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def main():
    import joblib
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Feature names ──────────────────────────────────────────────────
    feature_names = get_feature_names()
    save_feature_names(feature_names)
    n_features = len(feature_names)
    n_classes  = len(TARGET_EMOTIONS)
    logger.info(f"Feature dims: {n_features}  |  Classes: {n_classes}  →  {TARGET_EMOTIONS}")

    # ── Well-separated synthetic data ──────────────────────────────────
    # Use 300 samples per class (1500 total) with strong cluster separation
    X, y_raw = make_separated_dataset(
        n_per_class=300,
        n_features=n_features,
        n_classes=n_classes,
        separation=7.0,   # <-- key fix: wide separation for high confidence
        rng=rng,
    )
    logger.info(f"Synthetic dataset: {X.shape}  |  label distribution: {np.bincount(y_raw)}")

    # ── Label encoder ──────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(TARGET_EMOTIONS)
    # Map raw int labels → encoded emotion indices
    y = np.array([le.transform([TARGET_EMOTIONS[i % n_classes]])[0] for i in y_raw])

    # ── Scaler + SVM ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(
        kernel="rbf",
        C=10,           # higher C → tighter margins → higher peak confidence
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_scaled, y)

    # Quick cross-val sanity check
    cv_acc = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy")
    logger.info(f"Cross-val accuracy: {cv_acc.mean():.2%} ± {cv_acc.std():.2%}")

    # ── Save artefacts ─────────────────────────────────────────────────
    joblib.dump(clf,    BEST_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le,     LABEL_ENCODER_FILE)
    joblib.dump(clf,    MODELS_DIR / "svm_rbf_model.pkl")

    elapsed = time.time() - t0
    logger.info(f"✅  Demo model saved   → {BEST_MODEL_FILE}")
    logger.info(f"✅  Scaler saved       → {SCALER_FILE}")
    logger.info(f"✅  Label encoder      → {LABEL_ENCODER_FILE}")
    logger.info(f"⏱️  Build time         : {elapsed:.1f}s")
    logger.info("")
    logger.info("⚠️  This is a DEMO model trained on SYNTHETIC data.")
    logger.info("    Predictions will be plausible but NOT clinically accurate.")
    logger.info("    Download RAVDESS and run training/train_model.py for real accuracy.")


if __name__ == "__main__":
    main()
