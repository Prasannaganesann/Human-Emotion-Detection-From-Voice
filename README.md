<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

<h1 align="center">🎙️ Human Emotion Detection from Voice</h1>

<p align="center">
  <b>A production-grade AI system</b> that detects human emotions from voice recordings using<br>
  advanced audio signal processing, multi-model machine learning, and an interactive dashboard.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-training-pipeline">Training</a> •
  <a href="#-rest-api">API</a> •
  <a href="#-testing">Testing</a>
</p>

---

## ✨ Features

| Category | Details |
|:---|:---|
| **Audio Input** | File upload (WAV / MP3 / FLAC / OGG) · Live microphone recording · Batch multi-file analysis |
| **Preprocessing** | Silence trimming · Spectral noise reduction · Peak normalization · Fixed-duration padding |
| **Feature Extraction** | MFCC (40) + Δ + ΔΔ · Chroma STFT · Mel Spectrogram · Spectral Contrast · ZCR · RMS · Rolloff · Tonnetz |
| **ML Models** | SVM (RBF) · Random Forest · XGBoost · Gradient Boosting — with SMOTE oversampling & 5-fold CV |
| **Smart Labels** | Low-confidence predictions are flagged as ⚠️ **Uncertain** instead of forcing a hard label |
| **Evaluation** | Accuracy · Precision · Recall · F1 (weighted & macro) · Confusion Matrix · Cross-validation |
| **Dashboard** | Dark glassmorphism UI · Animated badges · Interactive Plotly charts · Emotion trend over time |
| **History** | SQLite-backed session tracking · Session-scoped or global clear · Aggregate analytics dashboard |
| **REST API** | FastAPI backend with `/predict`, `/predict/batch`, `/history`, `/stats`, `/trend` endpoints |
| **Testing** | pytest suite covering preprocessing, feature extraction, and model inference |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audio Input                              │
│              (File Upload / Microphone / Batch)                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Preprocessing Pipeline                                         │
│  ┌──────────┐  ┌─────────────┐  ┌───────────┐  ┌────────────┐  │
│  │  Load &   │→│   Trim      │→│  Spectral  │→│  Normalize  │  │
│  │  Resample │  │  Silence    │  │  Denoise   │  │  & Pad/Trim│  │
│  └──────────┘  └─────────────┘  └───────────┘  └────────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Feature Extraction (1104-dimensional vector per clip)          │
│  MFCC+Δ+ΔΔ │ Chroma │ Mel Spec │ Contrast │ ZCR│RMS│Roll│Tonn │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  ML Inference  (SVM / RF / XGBoost / GBM)                      │
│  StandardScaler → Model.predict_proba → LabelEncoder.inverse   │
│  Confidence < 35% → "Uncertain" label                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Output: Emotion + Confidence + Probabilities + Visualizations  │
│  → Streamlit Dashboard  /  FastAPI JSON  /  SQLite History      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── app/
│   ├── streamlit_app.py        # Main Streamlit UI (entry point)
│   └── api.py                  # FastAPI REST backend
├── data/
│   ├── raw/                    # RAVDESS audio files (user-provided)
│   ├── processed/              # Generated feature CSVs
│   └── build_dataset.py        # RAVDESS → feature CSV builder
├── models/
│   └── predictor.py            # Inference engine (model + scaler + encoder)
├── training/
│   ├── train_model.py          # Multi-model training pipeline
│   └── evaluate.py             # Standalone evaluation & charts
├── utils/
│   ├── preprocessing.py        # Audio preprocessing pipeline
│   ├── feature_extraction.py   # 1104-dim feature extraction
│   ├── visualizations.py       # Plotly chart library
│   └── database.py             # SQLite session history
├── tests/
│   └── test_core.py            # pytest suite
├── saved_models/               # Trained .pkl files (auto-created)
├── logs/                       # Evaluation outputs (auto-created)
├── config.py                   # Central configuration
├── generate_demo_model.py      # Quick-start synthetic model
├── requirements.txt            # All dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Prasannaganesann/Human-Emotion-Detection-From-Voice.git
cd Human-Emotion-Detection-From-Voice

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Generate Demo Model

This creates a synthetic model so you can run the app immediately — no dataset download needed:

```bash
python generate_demo_model.py
```

### 3. Launch the App

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** → Upload an audio file or record from your microphone → Click **Analyse Emotion**.

---

## 🏋️ Training Pipeline

### Step 1 — Download RAVDESS

Download the [RAVDESS Speech Audio Dataset](https://zenodo.org/record/1188976) and place it at:

```
data/raw/RAVDESS/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

### Step 2 — Build Feature Dataset

```bash
python data/build_dataset.py --data_dir data/raw/RAVDESS --out data/processed/features.csv
```

### Step 3 — Train Models

```bash
python training/train_model.py --features data/processed/features.csv
```

This trains **SVM · Random Forest · XGBoost · Gradient Boosting**, runs 5-fold stratified cross-validation, applies SMOTE oversampling for class balance, and saves the best model automatically.

### Step 4 — Evaluate

```bash
python training/evaluate.py --features data/processed/features.csv
```

Generates confusion matrices, per-class F1 charts, and a model comparison JSON in `logs/`.

---

## 🌐 REST API

### Start Server

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check |
| `GET` | `/models` | List available trained models |
| `POST` | `/predict` | Upload single audio → emotion prediction |
| `POST` | `/predict/batch` | Upload multiple audio files → batch results |
| `GET` | `/history?limit=50` | Prediction history (filterable by session) |
| `GET` | `/stats` | Aggregate emotion statistics |
| `GET` | `/trend?n=50` | Last N predictions for trend charts |

**Interactive docs:** http://localhost:8000/docs

### Example

```bash
# Single prediction
curl -X POST http://localhost:8000/predict -F "file=@audio.wav"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@audio1.wav" -F "files=@audio2.wav"
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

The test suite covers:
- **Preprocessing** — normalize, pad/trim, silence trimming, spectral denoising
- **Feature Extraction** — vector shape, dtype, finite values, name consistency
- **Predictor** — model loading, missing file handling, end-to-end inference

---

## 🎭 Supported Emotions

| RAVDESS Code | Emotion | Emoji |
|:---:|:---|:---:|
| 01 | Neutral | 😐 |
| 03 | Happy | 😊 |
| 04 | Sad | 😢 |
| 05 | Angry | 😠 |
| 06 | Fearful | 😨 |

> Predictions with confidence below **35%** are automatically labeled **❓ Uncertain**.

---

## 🧠 Feature Extraction Details

| Feature Group | Dimensions | Description |
|:---|:---:|:---|
| MFCC + Δ + ΔΔ | 40 × 3 × 4 = 480 | Mel-frequency cepstral coefficients with derivatives |
| Chroma STFT | 12 × 4 = 48 | Pitch class energy distribution |
| Mel Spectrogram | 128 × 4 = 512 | Log-scale mel filterbank energies |
| Spectral Contrast | 7 × 4 = 28 | Peak-valley contrast across frequency bands |
| Zero-Crossing Rate | 1 × 4 = 4 | Speech vs. noise indicator |
| RMS Energy | 1 × 4 = 4 | Signal loudness |
| Spectral Rolloff | 1 × 4 = 4 | Frequency concentration point |
| Tonnetz | 6 × 4 = 24 | Tonal centroid features |
| **Total** | **1,104** | Per-clip flat feature vector |

Each feature group is aggregated as **[mean, std, min, max]** across time frames.

---

## 📊 Model Performance (RAVDESS)

| Model | Accuracy | F1 (Weighted) |
|:---|:---:|:---:|
| **SVM (RBF)** | ~87% | ~86% |
| Random Forest | ~84% | ~83% |
| XGBoost | ~86% | ~85% |
| Gradient Boosting | ~83% | ~82% |

*Results vary with dataset size, preprocessing parameters, and class balance.*

---

## ⚙️ Configuration

All tunable parameters live in [`config.py`](config.py):

```python
SAMPLE_RATE     = 22050       # Hz
DURATION        = 3.0         # Seconds per clip
N_MFCC          = 40          # MFCC coefficients
N_MELS          = 128         # Mel filterbank bins
TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:---|:---|
| **ML** | Scikit-learn · XGBoost · imbalanced-learn · NumPy · Pandas |
| **Audio** | Librosa · SoundFile · SoundDevice |
| **Web** | Streamlit · Plotly · FastAPI · Uvicorn |
| **Storage** | SQLite · Joblib |
| **Testing** | pytest |

---

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub (this repo)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file**: `app/streamlit_app.py`
4. Add `requirements.txt` at root

> **Note:** Commit a pre-trained model or include `generate_demo_model.py` in a startup script.

---

## 👤 Author

**Prasanna Ganesan**

---

## 📄 License

MIT License — free for personal and commercial use.
