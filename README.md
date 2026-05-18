<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-6.0+-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e.svg" />
</p>

<h1 align="center">🎙️ VoiceEmo — Human Emotion Detection from Voice</h1>

<p align="center">
  <b>A production-grade AI system</b> that detects human emotions from voice recordings using<br>
  advanced audio signal processing, multi-model machine learning, and an interactive analytics dashboard.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-training-pipeline">Training</a> •
  <a href="#-rest-api">API</a> •
  <a href="#-testing">Testing</a>
</p>

> ⚠️ **Demo Mode Notice:** The quick-start demo model is trained on **synthetic data** for immediate testing.
> It produces plausible-looking predictions but is **not clinically accurate**.
> For real results, train on the [RAVDESS dataset](#-training-pipeline).

---

## ✨ Features

| Category | Details |
|:---|:---|
| **Audio Input** | File upload (WAV · FLAC · OGG) · MP3/M4A via ffmpeg fallback · Live microphone recording · Batch multi-file analysis |
| **Preprocessing** | Silence trimming · Spectral noise reduction · Peak normalization · Fixed-duration padding |
| **Feature Extraction** | MFCC (40) + Δ + ΔΔ · Chroma STFT · Mel Spectrogram · Spectral Contrast · ZCR · RMS · Rolloff · Tonnetz — **1,104-dim vector** |
| **ML Models** | SVM (RBF) · Random Forest · XGBoost · Gradient Boosting — with SMOTE oversampling & 5-fold CV |
| **Confidence** | Full probability distribution · Top-3 predictions with scores · Low-confidence predictions flagged as ⚠️ Uncertain |
| **Inference Metadata** | Emotion + confidence % + inference time (ms) + model version on every prediction |
| **Dashboard** | Dark glassmorphism UI · Animated badges · Interactive Plotly 6.x charts · Emotion trend over time |
| **History** | SQLite-backed session tracking · Search / filter / paginate · CSV export · Aggregate analytics |
| **REST API** | FastAPI backend with `/predict`, `/predict/batch`, `/history`, `/stats`, `/trend` endpoints |
| **Testing** | pytest suite — 15 tests covering preprocessing, feature extraction, and model inference |

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
│  Preprocessing Pipeline                                          │
│  ┌──────────┐  ┌─────────────┐  ┌───────────┐  ┌────────────┐   │
│  │  Load &  │→ │   Trim      │→ │  Spectral  │→ │  Normalize │   │
│  │  Resample│  │  Silence    │  │  Denoise   │  │  & Pad/Trim│   │
│  └──────────┘  └─────────────┘  └───────────┘  └────────────┘   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Feature Extraction  (1,104-dimensional vector per clip)         │
│  MFCC+Δ+ΔΔ │ Chroma │ Mel Spec │ Contrast │ ZCR│RMS│Roll│Tonn   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  ML Inference  (SVM / RF / XGBoost / GBM)                       │
│  StandardScaler → Model.predict_proba → LabelEncoder.inverse    │
│  Confidence < 30% → "Uncertain" flag + reason                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Output: Emotion + Confidence + Top-K + Inference Time (ms)     │
│  → Streamlit Dashboard  /  FastAPI JSON  /  SQLite History      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── app/
│   ├── streamlit_app.py        # Main Streamlit UI (entry point)
│   ├── styles.py               # CSS design token system
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
│   ├── feature_extraction.py   # 1,104-dim feature extraction
│   ├── visualizations.py       # Plotly 6.x chart library
│   └── database.py             # SQLite session history (search/filter/export)
├── tests/
│   └── test_core.py            # pytest suite (15 tests)
├── saved_models/               # Trained .pkl files (auto-created)
├── logs/                       # Evaluation outputs (auto-created)
├── config.py                   # Central configuration
├── generate_demo_model.py      # Quick-start synthetic demo model
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

This creates a **synthetic demo model** so you can run the app immediately — no dataset download needed.

> ⚠️ The demo model is trained on synthetic data. Predictions are illustrative only.
> See the [Training Pipeline](#-training-pipeline) section to train on real audio data.

```bash
python generate_demo_model.py
```

Output:
```
INFO  Feature dims: 1104  |  Classes: 5  →  ['neutral', 'happy', 'sad', 'angry', 'fearful']
INFO  Cross-val accuracy: 100.00% ± 0.00%
INFO  ✅  Demo model saved   → saved_models/best_emotion_model.pkl
```

### 3. Launch the App

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** → Upload an audio file or record from your microphone → Click **⚡ Analyse Emotion**.

---

## 🖥️ App Features

### Detect Emotion Tab
- Upload WAV / FLAC / OGG files (MP3/M4A supported if ffmpeg is installed)
- Live microphone recording with real-time progress indicator
- Result card showing: predicted emotion · confidence % · top-3 predictions · inference time
- Full probability distribution chart (Plotly)
- Audio signal statistics (duration, RMS, sample rate)

### Batch Predict Tab
- Upload multiple audio files at once
- Per-file progress bar and status
- CSV export of all results (emotion, confidence, duration, inference ms)

### Audio Analysis Tab
- Waveform visualization
- Log-Mel spectrogram heatmap
- MFCC coefficient heatmap
- Feature vector statistics (1,104 dimensions)

### History Tab
- Full prediction history with search, emotion filter, minimum confidence filter
- Paginated results (10 / 20 / 50 per page)
- CSV export of all records
- Emotion trend chart

### Dashboard Tab
- Total predictions · Dominant emotion · Weighted average confidence · Peak confidence
- Emotion distribution donut chart
- Prediction count bar chart per emotion
- Confidence trend over time
- Confidence score histogram
- Per-emotion statistics table

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

This trains **SVM · Random Forest · XGBoost · Gradient Boosting**, runs 5-fold stratified cross-validation, applies SMOTE oversampling for class balance, and saves the best model automatically to `saved_models/`.

### Step 4 — Evaluate

```bash
python training/evaluate.py --features data/processed/features.csv
```

Generates confusion matrices, per-class F1 charts, and a `model_comparison.json` in `logs/` which powers the Dashboard comparison chart.

---

## 🌐 REST API

### Start Server

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET`  | `/health` | Health check |
| `GET`  | `/models` | List available trained model files |
| `POST` | `/predict` | Upload single audio → full emotion prediction |
| `POST` | `/predict/batch` | Upload multiple audio files → batch results |
| `GET`  | `/history?limit=50` | Prediction history (filterable by session) |
| `GET`  | `/stats` | Aggregate emotion statistics |
| `GET`  | `/trend?n=50` | Last N predictions for trend charts |

**Interactive docs:** http://localhost:8000/docs  
**ReDoc:** http://localhost:8000/redoc

### Example Response (`/predict`)

```json
{
  "emotion": "happy",
  "confidence": 0.8723,
  "confidence_pct": "87.2%",
  "is_uncertain": false,
  "uncertainty_reason": null,
  "emoji": "😊",
  "color": "#F59E0B",
  "probabilities": { "neutral": 0.04, "happy": 0.87, "sad": 0.03, "angry": 0.02, "fearful": 0.04 },
  "top_k": [
    { "emotion": "happy",   "score": 0.8723, "pct": "87.2%" },
    { "emotion": "neutral", "score": 0.0412, "pct": "4.1%"  },
    { "emotion": "fearful", "score": 0.0381, "pct": "3.8%"  }
  ],
  "model_name": "best_emotion_model",
  "model_version": "v2.0",
  "inference_time_ms": 42.3,
  "audio_info": { "duration_s": 3.0, "sample_rate": 22050, "rms": 0.1823 }
}
```

### cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
     -F "file=@audio.wav"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@audio1.wav" \
     -F "files=@audio2.wav"
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

**15 tests** covering:

| Test Class | Coverage |
|:---|:---|
| `TestPreprocessing` | normalize, pad/trim, silence trimming, spectral denoising, audio info |
| `TestFeatureExtraction` | vector shape, dtype (float32), finite values, name length consistency |
| `TestPredictor` | missing file handling, model list, end-to-end inference, result keys |

Run with coverage:

```bash
python -m pytest tests/ -v --tb=short
```

---

## 🎭 Supported Emotions

| RAVDESS Code | Emotion | Emoji |
|:---:|:---|:---:|
| 01 | Neutral | 😐 |
| 03 | Happy | 😊 |
| 04 | Sad | 😢 |
| 05 | Angry | 😠 |
| 06 | Fearful | 😨 |

> Predictions with confidence below **30%** are automatically flagged as **⚠️ Uncertain** with a human-readable reason.

---

## 🧠 Feature Extraction Details

| Feature Group | Dimensions | Description |
|:---|:---:|:---|
| MFCC + Δ + ΔΔ | 40 × 3 × 4 = 480 | Mel-frequency cepstral coefficients with 1st & 2nd derivatives |
| Chroma STFT | 12 × 4 = 48 | Pitch class energy distribution |
| Mel Spectrogram | 128 × 4 = 512 | Log-scale mel filterbank energies |
| Spectral Contrast | 7 × 4 = 28 | Peak-valley contrast across frequency bands |
| Zero-Crossing Rate | 1 × 4 = 4 | Speech vs. noise indicator |
| RMS Energy | 1 × 4 = 4 | Signal loudness |
| Spectral Rolloff | 1 × 4 = 4 | Frequency concentration point |
| Tonnetz | 6 × 4 = 24 | Tonal centroid features |
| **Total** | **1,104** | Flat float32 vector per audio clip |

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
*The demo model uses synthetic data and will not match these numbers.*

---

## ⚙️ Configuration

All tunable parameters live in [`config.py`](config.py):

```python
SAMPLE_RATE       = 22050       # Hz
DURATION          = 3.0         # Seconds per clip
N_MFCC            = 40          # MFCC coefficients
N_MELS            = 128         # Mel filterbank bins
RECORDING_DURATION = 5          # Default microphone recording length (s)
TARGET_EMOTIONS   = ["neutral", "happy", "sad", "angry", "fearful"]
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:---|:---|
| **ML** | Scikit-learn · XGBoost · imbalanced-learn · NumPy · Pandas |
| **Audio** | Librosa · SoundFile · SoundDevice |
| **Web** | Streamlit · Plotly 6.x · FastAPI · Uvicorn |
| **Storage** | SQLite · Joblib |
| **Testing** | pytest |

---

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub (this repo is already configured)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file path**: `app/streamlit_app.py`
4. Ensure `requirements.txt` is at the root

> **Important:** Add a startup script or pre-commit the demo model PKL files.
> Streamlit Cloud does not persist generated files between deploys.
> Add a `packages.txt` with `ffmpeg` if you need MP3/M4A support.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python generate_demo_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🐛 Known Limitations

- **Demo model** — synthetic training data; replace with RAVDESS for real accuracy
- **MP3 / M4A** — requires `ffmpeg` installed on the system; WAV/FLAC/OGG work natively
- **Microphone recording** — uses `sounddevice` which accesses system audio directly; not supported on Streamlit Cloud (upload audio files instead)
- **5 emotions only** — current label set is Neutral · Happy · Sad · Angry · Fearful (RAVDESS subset)

---

## 👤 Author

**Prasanna Ganesan**  
[GitHub](https://github.com/Prasannaganesann)

---

## 📄 License

MIT License — free for personal and commercial use.
