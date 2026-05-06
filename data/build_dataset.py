"""
build_dataset.py
================
Scans the RAVDESS dataset folder and builds a feature matrix CSV
ready for model training.

RAVDESS file naming convention:
  03-01-05-01-02-01-12.wav
  └─ Modality─┘  └─ Emotion ─┘

  Positions (1-indexed):
  1 – Modality (01=full-AV, 02=video-only, 03=audio-only)
  2 – Vocal channel (01=speech, 02=song)
  3 – Emotion (01=neutral, 02=calm, ..., 08=surprised)
  4 – Emotional intensity
  5 – Statement
  6 – Repetition
  7 – Actor

Usage:
    python data/build_dataset.py --data_dir data/raw/RAVDESS --out data/processed/features.csv
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import RAVDESS_EMOTION_MAP, TARGET_EMOTIONS, PROCESSED_DIR
from utils.preprocessing import preprocess_audio
from utils.feature_extraction import extract_features, get_feature_names, save_feature_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RAVDESS Parser
# ─────────────────────────────────────────────

def parse_ravdess_label(filename: str) -> str | None:
    """
    Parse the emotion label from a RAVDESS filename.

    Returns the emotion string (e.g. 'happy') or None if not in TARGET_EMOTIONS.
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None

    emotion_code = parts[2]                    # 3rd field
    emotion = RAVDESS_EMOTION_MAP.get(emotion_code)

    if emotion in TARGET_EMOTIONS:
        return emotion
    return None


# ─────────────────────────────────────────────
#  Dataset Builder
# ─────────────────────────────────────────────

def build_dataset(data_dir: str, out_csv: str) -> pd.DataFrame:
    """
    Walk the RAVDESS directory tree, extract features for every audio file,
    and save to a CSV.

    Parameters
    ----------
    data_dir : str
        Root folder containing RAVDESS actor subdirectories.
    out_csv : str
        Output path for the features CSV.

    Returns
    -------
    pd.DataFrame : feature matrix with a 'label' column.
    """
    data_path = Path(data_dir)
    wav_files  = sorted(data_path.rglob("*.wav"))

    if not wav_files:
        logger.error(f"No .wav files found under {data_path}. "
                     "Please download the RAVDESS dataset first.")
        sys.exit(1)

    logger.info(f"Found {len(wav_files)} .wav files in {data_path}")

    feature_names = get_feature_names()
    save_feature_names(feature_names)

    rows = []
    skipped = 0

    for wav in tqdm(wav_files, desc="Extracting features", unit="file"):
        label = parse_ravdess_label(wav.name)
        if label is None:
            skipped += 1
            continue

        try:
            y, sr = preprocess_audio(file_path=str(wav),
                                     apply_noise_reduction=False)
            feat = extract_features(y, sr)
            row  = dict(zip(feature_names, feat))
            row["label"] = label
            rows.append(row)
        except Exception as e:
            logger.warning(f"Skipped {wav.name}: {e}")
            skipped += 1

    logger.info(f"Processed {len(rows)} samples, skipped {skipped}")

    df = pd.DataFrame(rows)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved feature matrix → {out_path}  shape={df.shape}")

    # Print class distribution
    dist = df["label"].value_counts()
    logger.info(f"\nClass distribution:\n{dist.to_string()}")

    return df


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAVDESS feature dataset")
    parser.add_argument("--data_dir", default="data/raw/RAVDESS",
                        help="Path to RAVDESS audio folder")
    parser.add_argument("--out",      default="data/processed/features.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    build_dataset(args.data_dir, args.out)
