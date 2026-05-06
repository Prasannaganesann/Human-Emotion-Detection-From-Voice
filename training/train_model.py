"""
train_model.py
==============
Trains SVM, Random Forest, and XGBoost classifiers on the extracted
feature matrix. Evaluates all models, saves per-model metrics, and
persists the best model + scaler for inference.

Usage:
    python training/train_model.py
    python training/train_model.py --features data/processed/features.csv
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    BEST_MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE
)


# ─────────────────────────────────────────────
#  Model Definitions
# ─────────────────────────────────────────────

def get_models() -> dict:
    """Return a dict of named classifiers to benchmark."""
    return {
        "SVM_RBF": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


# ─────────────────────────────────────────────
#  Training Pipeline
# ─────────────────────────────────────────────

def load_features(csv_path: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load features CSV and encode labels."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape}  |  labels: {df['label'].unique()}")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    logger.info(f"Classes: {le.classes_}")

    return X, y, le


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   le: LabelEncoder) -> dict:
    """Compute detailed metrics on the test split."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy":     round(acc,  4),
        "precision":    round(prec, 4),
        "recall":       round(rec,  4),
        "f1_weighted":  round(f1_w, 4),
        "f1_macro":     round(f1_m, 4),
        "report":       report,
        "confusion_matrix": cm,
        "classes":      le.classes_.tolist(),
    }


def train(csv_path: str, output_dir: str = None) -> dict:
    """
    Full training pipeline:
      1. Load & scale features
      2. SMOTE oversampling
      3. Train multiple classifiers
      4. Cross-validation + held-out test evaluation
      5. Save best model + scaler + label encoder + results JSON

    Returns
    -------
    dict : {model_name: metrics} for all trained models.
    """
    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    X, y, le = load_features(csv_path)
    logger.info(f"Feature matrix: {X.shape}")

    # ── 2. Train/test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── 3. Scale ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 4. SMOTE (on training set only) ───────────────────────────────
    try:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        logger.info(f"After SMOTE: {X_train_s.shape}")
    except Exception as e:
        logger.warning(f"SMOTE skipped: {e}")

    # ── 5. Train & evaluate each model ────────────────────────────────
    models_dict  = get_models()
    all_results  = {}
    trained      = {}

    for name, clf in models_dict.items():
        logger.info(f"\n{'='*60}\nTraining {name}\n{'='*60}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        cv_res = cross_validate(
            clf, X_train_s, y_train, cv=cv,
            scoring=["accuracy", "f1_weighted"],
            return_train_score=False, n_jobs=-1,
        )
        cv_acc = cv_res["test_accuracy"].mean()
        cv_f1  = cv_res["test_f1_weighted"].mean()
        logger.info(f"CV Accuracy: {cv_acc:.4f}  |  CV F1: {cv_f1:.4f}")

        # Final fit on full training set
        clf.fit(X_train_s, y_train)

        # Test-set evaluation
        metrics = evaluate_model(clf, X_test_s, y_test, le)
        metrics["cv_accuracy"] = round(float(cv_acc), 4)
        metrics["cv_f1"]       = round(float(cv_f1),  4)

        logger.info(f"Test  Accuracy: {metrics['accuracy']:.4f}  |  "
                    f"F1: {metrics['f1_weighted']:.4f}")

        all_results[name] = metrics
        trained[name]     = clf

        # Save individual model
        model_path = output_dir / f"{name.lower()}_model.pkl"
        joblib.dump(clf, model_path)
        logger.info(f"Saved {name} → {model_path}")

    # ── 6. Pick best model ────────────────────────────────────────────
    best_name = max(all_results, key=lambda n: all_results[n]["f1_weighted"])
    best_clf  = trained[best_name]
    logger.info(f"\n🏆  Best model: {best_name}  "
                f"(F1={all_results[best_name]['f1_weighted']:.4f})")

    joblib.dump(best_clf, BEST_MODEL_FILE)
    joblib.dump(scaler,   SCALER_FILE)
    joblib.dump(le,       LABEL_ENCODER_FILE)

    # ── 7. Persist results JSON ───────────────────────────────────────
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump({"best_model": best_name, "results": all_results}, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    return all_results


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classifiers")
    parser.add_argument("--features", default="data/processed/features.csv",
                        help="Path to feature CSV")
    parser.add_argument("--output",   default=str(MODELS_DIR),
                        help="Directory to save trained models")
    args = parser.parse_args()

    results = train(args.features, args.output)

    print("\n" + "─" * 60)
    print("TRAINING SUMMARY")
    print("─" * 60)
    for name, m in results.items():
        print(f"{name:25s}  Acc={m['accuracy']:.4f}  F1={m['f1_weighted']:.4f}")
    print("─" * 60)
