"""
evaluate.py
===========
Standalone evaluation script.
Loads a trained model and produces:
  - Full classification report (console)
  - Confusion matrix PNG
  - Per-class accuracy bar chart PNG
  - Model comparison chart (if multiple model files exist)

Usage:
    python training/evaluate.py
    python training/evaluate.py --model saved_models/svm_rbf_model.pkl --features data/processed/features.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    MODELS_DIR, SCALER_FILE, LABEL_ENCODER_FILE,
    BEST_MODEL_FILE, RANDOM_STATE, TEST_SIZE, LOGS_DIR
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Evaluation Helpers
# ─────────────────────────────────────────────

def load_test_data(csv_path: str):
    """Load features CSV and return scaled X_test, y_test, and label names."""
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    le     = joblib.load(LABEL_ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)

    y      = le.transform(y_raw)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_test_s = scaler.transform(X_test)

    return X_test_s, y_test, le.classes_


def plot_and_save_confusion_matrix(cm: np.ndarray, labels: list[str],
                                    model_name: str) -> str:
    """Render and save a seaborn confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()

    out_path = str(LOGS_DIR / f"confusion_matrix_{model_name.lower()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {out_path}")
    return out_path


def plot_per_class_accuracy(report: dict, model_name: str) -> str:
    """Bar chart of per-class F1 scores."""
    classes = [k for k in report.keys()
               if k not in ("accuracy", "macro avg", "weighted avg")]
    f1_scores = [report[c]["f1-score"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([c.capitalize() for c in classes], f1_scores,
                  color=sns.color_palette("husl", len(classes)))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1 – {model_name}", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()

    out_path = str(LOGS_DIR / f"per_class_f1_{model_name.lower()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Per-class F1 chart saved → {out_path}")
    return out_path


def evaluate_all_models(csv_path: str) -> dict:
    """
    Evaluate all saved .pkl models in saved_models/ against the test set.

    Returns
    -------
    dict : {model_name: metrics}
    """
    X_test, y_test, label_names = load_test_data(csv_path)
    results = {}

    model_files = list(MODELS_DIR.glob("*_model.pkl"))
    if not model_files:
        logger.warning("No model files found in saved_models/")
        return {}

    for model_path in model_files:
        name = model_path.stem.replace("_model", "").replace("_", " ").title()
        logger.info(f"\nEvaluating: {name}")

        clf = joblib.load(model_path)
        y_pred = clf.predict(X_test)

        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        # Print to console
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        print(classification_report(y_test, y_pred,
                                     target_names=label_names,
                                     zero_division=0))

        # Save plots
        plot_and_save_confusion_matrix(cm, label_names, name)
        plot_per_class_accuracy(report, name)

        results[name] = {
            "accuracy":    round(report["accuracy"], 4),
            "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
            "f1_macro":    round(report["macro avg"]["f1-score"], 4),
            "report":      report,
        }

    # Save comparison JSON
    comp_path = LOGS_DIR / "model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Comparison saved → {comp_path}")

    # Print summary table
    print("\n" + "─" * 60)
    print(f"{'Model':<25}  {'Accuracy':>10}  {'F1 Weighted':>12}")
    print("─" * 60)
    for n, m in results.items():
        print(f"{n:<25}  {m['accuracy']:>10.4f}  {m['f1_weighted']:>12.4f}")
    print("─" * 60)

    return results


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained emotion models")
    parser.add_argument("--features", default="data/processed/features.csv",
                        help="Feature CSV path")
    args = parser.parse_args()
    evaluate_all_models(args.features)
