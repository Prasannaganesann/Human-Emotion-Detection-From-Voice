"""
streamlit_app.py  –  Human Emotion Detection from Voice
=========================================================
Production-grade Streamlit UI with:
  • File upload  +  live microphone recording
  • Real-time emotion prediction
  • Probability distribution chart
  • Waveform / Mel-spectrogram / MFCC visualizations
  • Emotion trend over time
  • Session history panel
  • Model switcher
  • Dashboard tab with aggregate stats
"""

import io
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    SAMPLE_RATE, RECORDING_DURATION, EMOTION_COLORS,
    EMOTION_EMOJIS, MODELS_DIR, LOGS_DIR
)
from models.predictor import EmotionPredictor
from utils.database import (
    init_db, save_prediction, get_predictions,
    get_emotion_stats, get_trend_data, clear_history
)
from utils.preprocessing import preprocess_audio, get_audio_info
from utils.visualizations import (
    plot_waveform, plot_mel_spectrogram, plot_mfcc,
    plot_emotion_probabilities, plot_emotion_trend,
    plot_emotion_distribution, plot_model_comparison
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceEmo · Emotion Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS – dark glassmorphism theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); }

/* Glassmorphism cards */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
}

/* Emotion result badge */
.emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 28px;
    border-radius: 50px;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
    50%      { box-shadow: 0 0 0 14px rgba(99,102,241,0); }
}

/* Confidence ring */
.confidence-ring {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03); border-radius: 12px; }
.stTabs [data-baseweb="tab"]      { border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"]    { background: rgba(99,102,241,0.3) !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(99,102,241,0.5);
    border-radius: 14px;
}

/* History table */
.history-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; border-radius: 10px;
    background: rgba(255,255,255,0.04);
    margin-bottom: 8px;
    border-left: 3px solid var(--border-color);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Initialise DB & Session State
# ─────────────────────────────────────────────
init_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None


# ─────────────────────────────────────────────
#  Load Predictor (cached per model path)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(_model_path: str):
    """Cache one predictor instance per unique model path."""
    p = EmotionPredictor()
    p.load(_model_path)
    return p


def get_predictor():
    mp = st.session_state.get("model_path")
    if not mp:
        return None
    try:
        return load_predictor(mp)
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ **VoiceEmo**")
    st.markdown("*Human Emotion Detection from Voice*")
    st.divider()

    # Model selector
    model_files = list(MODELS_DIR.glob("*_model.pkl"))
    if model_files:
        model_names = {p.stem.replace("_model","").replace("_"," ").title(): str(p)
                       for p in model_files}
        selected_name = st.selectbox("🤖 Active Model", list(model_names.keys()))
        st.session_state.model_path = model_names[selected_name]
    else:
        st.warning("⚠️ No trained model found.\nRun `training/train_model.py` first.")

    st.divider()

    # Session info
    st.markdown(f"**Session:** `{st.session_state.session_id}`")
    pred_count = len(get_predictions(session_id=st.session_state.session_id))
    st.metric("Predictions this session", pred_count)

    st.divider()

    clear_scope = st.radio("Clear scope", ["This session", "All sessions"],
                           horizontal=True, label_visibility="collapsed")
    if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
        if clear_scope == "This session":
            clear_history(session_id=st.session_state.session_id)
            st.success("Session history cleared!")
        else:
            clear_history()
            st.success("All history cleared!")

    st.divider()
    st.markdown("**About**")
    st.caption("Features: MFCC · Chroma · Mel · ZCR · RMS\nModels: SVM · RF · XGBoost · GBM\nDataset: RAVDESS")


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px;">
  <h1 style="font-size:2.6rem; font-weight:800; background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
    🎙️ Human Emotion Detection
  </h1>
  <p style="color:#94a3b8; font-size:1.05rem; margin-top:6px;">
    Upload a voice clip or record live — discover the emotion within
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
#  Main Tabs
# ─────────────────────────────────────────────
tab_detect, tab_batch, tab_visualize, tab_history, tab_dashboard = st.tabs([
    "🎯 Detect Emotion", "📦 Batch Predict", "📊 Audio Analysis", "📜 History", "🧠 Dashboard"
])


# ══════════════════════════════════════════════
#  TAB 1 – Detect Emotion
# ══════════════════════════════════════════════
with tab_detect:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("### 📥 Audio Input")

        input_mode = st.radio(
            "Choose input method",
            ["📁 Upload File", "🎤 Record Microphone"],
            horizontal=True, label_visibility="collapsed"
        )

        audio_bytes_input = None
        recorded_array    = None

        # ── File upload ────────────────────────────────────
        if input_mode == "📁 Upload File":
            uploaded = st.file_uploader(
                "Drop a .wav / .mp3 file",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                label_visibility="collapsed",
            )
            if uploaded:
                audio_bytes_input = uploaded.getvalue()
                st.audio(audio_bytes_input)
                st.caption(f"📄 {uploaded.name}  ·  {len(audio_bytes_input)/1024:.1f} KB")

        # ── Microphone recording ───────────────────────────
        else:
            rec_dur = st.slider("Recording duration (s)", 2, 10,
                                RECORDING_DURATION, key="rec_dur")
            st.caption("⚠️ Make sure your microphone is connected and permitted.")

            if st.button("🔴 Start Recording", type="primary",
                         use_container_width=True):
                placeholder = st.empty()
                with placeholder.container():
                    st.info(f"🎙️ Recording for **{rec_dur}s** … speak now!")
                    progress = st.progress(0)
                    for i in range(rec_dur * 10):
                        time.sleep(0.1)
                        progress.progress((i + 1) / (rec_dur * 10))

                try:
                    audio_np = sd.rec(
                        int(rec_dur * SAMPLE_RATE),
                        samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32"
                    )
                    sd.wait()
                    recorded_array = audio_np.flatten()
                    st.session_state.audio_data = recorded_array

                    # Convert to bytes for playback
                    buf = io.BytesIO()
                    sf.write(buf, recorded_array, SAMPLE_RATE, format="WAV")
                    audio_bytes_input = buf.getvalue()
                    st.audio(audio_bytes_input, format="audio/wav")
                    placeholder.empty()
                    st.success("✅ Recording complete!")
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Recording failed: {e}")

        # ── Analyse button ─────────────────────────────────
        st.markdown("---")
        analyse_btn = st.button(
            "⚡ Analyse Emotion", type="primary",
            use_container_width=True,
            disabled=(audio_bytes_input is None and recorded_array is None)
        )

    # ── Result panel ──────────────────────────────────────
    with col_result:
        st.markdown("### 🎯 Result")

        predictor = get_predictor()

        if analyse_btn and predictor:
            with st.spinner("Analysing audio …"):
                try:
                    if recorded_array is not None:
                        result = predictor.predict_from_array(
                            recorded_array, SAMPLE_RATE
                        )
                        filename = "microphone_recording"
                    else:
                        result = predictor.predict_from_bytes(
                            audio_bytes_input, filename="uploaded_file"
                        )
                        filename = "uploaded_file"

                    st.session_state.last_result = result
                    st.session_state.audio_bytes = audio_bytes_input
                    st.session_state.audio_array = (
                        recorded_array
                        if recorded_array is not None
                        else None
                    )

                    # Store in DB
                    save_prediction(
                        emotion=result["emotion"],
                        confidence=result["confidence"],
                        probabilities=result["probabilities"],
                        model_name=result["model_name"],
                        filename=filename,
                        duration_s=result["audio_info"].get("duration_s"),
                        session_id=st.session_state.session_id,
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

        # Show last result
        result = st.session_state.last_result
        if result:
            emotion    = result["emotion"]
            confidence = result["confidence"]
            color      = result["color"]
            emoji      = result["emoji"]
            probs      = result["probabilities"]

            # Uncertain label
            uncertain_note = ""
            if result.get("is_uncertain"):
                uncertain_note = ('<div style="color:#f59e0b; font-size:0.85rem; '
                                  'margin-top:8px; font-weight:600;">'
                                  '⚠️ Low confidence — prediction may be unreliable</div>')

            # Emotion badge
            st.markdown(f"""
            <div style="text-align:center; padding:20px 0 10px;">
              <div class="emotion-badge" style="background:{color}22;
                   border:2px solid {color}; color:{color};
                   margin:0 auto; width:fit-content;">
                {emoji} {emotion.capitalize()}
              </div>
              <div class="confidence-ring" style="margin-top:16px;">
                {confidence*100:.1f}%
              </div>
              <div style="color:#94a3b8; font-size:0.9rem; margin-top:4px;">
                Confidence · {result['model_name'].replace('_',' ').title()}
              </div>
              {uncertain_note}
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            st.plotly_chart(
                plot_emotion_probabilities(probs, emotion),
                use_container_width=True
            )

            # Audio stats
            if result["audio_info"]:
                info = result["audio_info"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Duration", f"{info.get('duration_s',0):.2f}s")
                c2.metric("RMS Energy", f"{info.get('rms',0):.4f}")
                c3.metric("Sample Rate", f"{info.get('sample_rate',0)} Hz")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#4b5563;">
              <div style="font-size:4rem;">🎙️</div>
              <p>Upload or record audio, then click <strong>Analyse Emotion</strong></p>
            </div>
            """, unsafe_allow_html=True)

        if not predictor:
            st.warning("⚠️ No trained model loaded. Run `python generate_demo_model.py` first.")


# ══════════════════════════════════════════════
#  TAB – Batch Prediction
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📦 Batch Prediction")
    st.caption("Upload multiple audio files and get predictions for all of them at once.")

    batch_files = st.file_uploader(
        "Drop multiple audio files",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if batch_files and st.button("⚡ Analyse Batch", key="batch_btn", type="primary"):
        predictor = get_predictor()
        if not predictor:
            st.warning("No model loaded.")
        else:
            items = [(f.getvalue(), f.name) for f in batch_files]
            with st.spinner(f"Analysing {len(items)} files…"):
                results = predictor.predict_batch_bytes(items)

            for i, (f, res) in enumerate(zip(batch_files, results)):
                if "error" in res:
                    st.error(f"❌ {f.name}: {res['error']}")
                    continue
                color = res["color"]
                emoji = res["emoji"]
                emo = res["emotion"]
                conf = res["confidence"]
                uncertain = " ⚠️" if res.get("is_uncertain") else ""
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:14px; padding:12px 18px;
                     border-radius:12px; background:rgba(255,255,255,0.04);
                     border-left:4px solid {color}; margin-bottom:8px;">
                  <span style="font-size:1.5rem;">{emoji}</span>
                  <div style="flex:1;">
                    <strong style="color:{color};">{emo.capitalize()}{uncertain}</strong>
                    <span style="color:#94a3b8; font-size:0.85rem;"> · {f.name}</span>
                  </div>
                  <span style="font-weight:700; color:white;">{conf*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                # Save each to DB
                save_prediction(
                    emotion=emo, confidence=conf,
                    probabilities=res["probabilities"],
                    model_name=res["model_name"],
                    filename=f.name,
                    duration_s=res["audio_info"].get("duration_s"),
                    session_id=st.session_state.session_id,
                )
            st.success(f"✅ Batch complete — {len(results)} files analysed.")


# ══════════════════════════════════════════════
#  TAB 2 – Audio Analysis / Visualizations
# ══════════════════════════════════════════════
with tab_visualize:
    st.markdown("### 📊 Audio Signal Analysis")

    result = st.session_state.last_result
    if result is None:
        st.info("🔍 Analyse an audio clip first to see visualizations here.")
    else:
        # Reconstruct audio from bytes
        try:
            audio_bytes = st.session_state.get("audio_bytes")
            if audio_bytes:
                y, sr = preprocess_audio(audio_bytes=audio_bytes)
            else:
                y, sr = preprocess_audio(audio_array=st.session_state.get("audio_array"))

            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                st.plotly_chart(plot_mfcc(y, sr), use_container_width=True)

            with col_v2:
                st.plotly_chart(plot_mel_spectrogram(y, sr), use_container_width=True)

                # Feature importance (top MFCC values)
                from utils.feature_extraction import extract_features
                feat = extract_features(y, sr)
                st.markdown("**Feature Vector Statistics**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Feature Dims", len(feat))
                c2.metric("Mean Value",  f"{feat.mean():.4f}")
                c3.metric("Std Dev",     f"{feat.std():.4f}")

        except Exception as e:
            st.error(f"Visualization error: {e}")


# ══════════════════════════════════════════════
#  TAB 3 – History
# ══════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Prediction History")

    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        limit = st.slider("Show last N predictions", 5, 100, 20, key="hist_limit")
        history = get_predictions(limit=limit)

        if not history:
            st.info("No predictions recorded yet.")
        else:
            for rec in history:
                color = EMOTION_COLORS.get(rec["emotion"], "#94A3B8")
                emoji = EMOTION_EMOJIS.get(rec["emotion"], "🎙️")
                ts_str = rec["timestamp"].replace("T", " ")

                st.markdown(f"""
                <div class="history-row" style="--border-color:{color};">
                  <span style="font-size:1.4rem;">{emoji}</span>
                  <div style="flex:1;">
                    <strong style="color:{color};">{rec['emotion'].capitalize()}</strong>
                    &nbsp;·&nbsp;
                    <span style="color:#94a3b8; font-size:0.85rem;">{ts_str}</span>
                  </div>
                  <div style="text-align:right;">
                    <span style="font-weight:700; color:white;">
                      {rec['confidence']*100:.1f}%
                    </span><br>
                    <span style="color:#6b7280; font-size:0.75rem;">
                      {rec.get('filename','–')}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with col_h2:
        st.markdown("#### Emotion Trend")
        trend_data = get_trend_data(n=30)
        st.plotly_chart(plot_emotion_trend(trend_data), use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 – Dashboard
# ══════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 🧠 Analytics Dashboard")

    stats = get_emotion_stats()

    if not stats:
        st.info("📊 No data yet — make some predictions to see the dashboard.")
    else:
        total = sum(v["count"] for v in stats.values())
        dominant_emotion = max(stats, key=lambda k: stats[k]["count"])

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", total)
        k2.metric("Dominant Emotion",
                  f"{EMOTION_EMOJIS.get(dominant_emotion,'')} {dominant_emotion.capitalize()}")
        k3.metric("Unique Emotions", len(stats))
        avg_conf = np.mean([v["avg_confidence"] for v in stats.values()])
        k4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

        st.divider()

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.plotly_chart(plot_emotion_distribution(stats),
                            use_container_width=True)

        with col_d2:
            # Load model comparison if available
            comp_path = LOGS_DIR / "model_comparison.json"
            if comp_path.exists():
                with open(comp_path) as f:
                    comp_data = json.load(f)
                st.plotly_chart(plot_model_comparison(comp_data),
                                use_container_width=True)
            else:
                # Show emotion bar chart instead
                import plotly.express as px
                emotions = list(stats.keys())
                counts   = [stats[e]["count"] for e in emotions]
                colors   = [EMOTION_COLORS.get(e, "#94A3B8") for e in emotions]
                fig = px.bar(x=[e.capitalize() for e in emotions], y=counts,
                             color=emotions,
                             color_discrete_map={e: EMOTION_COLORS.get(e,"#94A3B8")
                                                 for e in emotions},
                             title="Emotion Count (All Sessions)",
                             template="plotly_dark")
                fig.update_layout(showlegend=False,
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0.15)",
                                  height=320)
                st.plotly_chart(fig, use_container_width=True)

        # Full trend chart
        st.markdown("#### 📈 Full Emotion Trend")
        trend_all = get_trend_data(n=100)
        st.plotly_chart(plot_emotion_trend(trend_all), use_container_width=True)

        # Detailed stats table
        st.markdown("#### 📋 Emotion Statistics Table")
        import pandas as pd
        df_stats = pd.DataFrame([
            {
                "Emotion":         e.capitalize(),
                "Count":           stats[e]["count"],
                "Avg Confidence":  f"{stats[e]['avg_confidence']*100:.1f}%",
                "Share":           f"{stats[e]['count']/total*100:.1f}%",
            }
            for e in stats
        ])
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
