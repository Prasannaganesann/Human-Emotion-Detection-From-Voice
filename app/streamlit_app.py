"""
streamlit_app.py — VoiceEmo v2
================================
Production-grade Streamlit UI for Human Emotion Detection from Voice.

Tabs:
  🎯 Detect   — upload or record, get rich prediction card
  📦 Batch    — multi-file with CSV export
  📊 Analysis — waveform / spectrogram / MFCC with explanations
  📜 History  — searchable, filterable, paginated history
  🧠 Dashboard — analytics, trend, distribution, confidence histogram
"""

import io, json, logging, sys, time, uuid
from pathlib import Path

import numpy as np
import streamlit as st
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.styles import CSS
from config import (
    SAMPLE_RATE, RECORDING_DURATION, EMOTION_COLORS, EMOTION_EMOJIS, MODELS_DIR, LOGS_DIR
)
from models.predictor import EmotionPredictor
from utils.database import (
    init_db, save_prediction, get_predictions, get_predictions_count,
    get_emotion_stats, get_trend_data, clear_history,
    get_distinct_emotions, export_predictions_csv,
)
from utils.preprocessing import preprocess_audio, get_audio_info
from utils.visualizations import (
    plot_waveform, plot_mel_spectrogram, plot_mfcc,
    plot_emotion_probabilities, plot_emotion_trend,
    plot_emotion_distribution, plot_model_comparison,
    plot_confidence_histogram,
)

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceEmo · AI Emotion Detection",
    page_icon="🎙️", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
init_db()
for key, default in [
    ("session_id",   str(uuid.uuid4())[:8]),
    ("last_result",  None),
    ("audio_bytes",  None),
    ("audio_array",  None),
    ("model_path",   None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Model loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str):
    p = EmotionPredictor()
    p.load(model_path)
    return p

def get_predictor():
    mp = st.session_state.get("model_path")
    if not mp:
        return None
    try:
        return load_predictor(mp)
    except FileNotFoundError:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 0 2px">
      <div style="font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.5px">🎙️ VoiceEmo</div>
      <div style="color:#64748b;font-size:.72rem;letter-spacing:.8px;text-transform:uppercase;margin-top:2px">AI EMOTION DETECTION</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Model selector
    model_files = list(MODELS_DIR.glob("*_model.pkl"))
    if model_files:
        model_map = {p.stem.replace("_model","").replace("_"," ").title(): str(p) for p in model_files}
        sel = st.selectbox("🤖 Active Model", list(model_map.keys()), label_visibility="visible")
        st.session_state.model_path = model_map[sel]
    else:
        st.error("⚠️ No model found.\nRun `python generate_demo_model.py`")

    st.divider()

    # Session info
    pred_count = len(get_predictions(session_id=st.session_state.session_id))
    st.markdown(f"""
    <div class="section-label">📡 Session</div>
    <div style="color:#c7d2fe;font-family:monospace;font-size:.88rem;font-weight:600;margin-bottom:8px">{st.session_state.session_id}</div>
    """, unsafe_allow_html=True)
    st.metric("Predictions", pred_count)
    st.divider()

    # Clear history
    scope = st.radio("Scope", ["This session", "All sessions"], horizontal=True, label_visibility="collapsed")
    if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
        sid = st.session_state.session_id if scope == "This session" else None
        clear_history(session_id=sid)
        st.session_state.last_result = None
        st.success("History cleared!")

    st.divider()
    st.markdown("""
    <div class="section-label">ℹ️ About</div>
    <div style="color:#64748b;font-size:.79rem;line-height:1.65">
      <b style="color:#a5b4fc">Features:</b> MFCC · Chroma · Mel · ZCR · RMS<br>
      <b style="color:#a5b4fc">Models:</b> SVM · RF · XGBoost<br>
      <b style="color:#a5b4fc">Dataset:</b> RAVDESS<br>
      <b style="color:#a5b4fc">Version:</b> v2.0
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:8px 0 4px">
  <h1 style="font-size:2.2rem;font-weight:800;margin:0;
             background:linear-gradient(135deg,#818cf8 0%,#a78bfa 45%,#c084fc 75%,#f472b6 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    🎙️ Human Emotion Detection
  </h1>
  <p style="color:#64748b;font-size:.95rem;margin-top:5px;font-weight:400">
    Analyse voice audio to detect emotional tone using AI
  </p>
</div>""", unsafe_allow_html=True)

tab_detect, tab_batch, tab_viz, tab_history, tab_dash = st.tabs([
    "🎯 Detect Emotion", "📦 Batch Predict", "📊 Audio Analysis", "📜 History", "🧠 Dashboard"
])


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — DETECT EMOTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_detect:
    col_in, col_res = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown("### 📥 Audio Input")
        mode = st.radio("Input", ["📁 Upload File", "🎤 Record Microphone"],
                        horizontal=True, label_visibility="collapsed")

        audio_bytes_input = None
        recorded_array    = None

        if mode == "📁 Upload File":
            up = st.file_uploader("Drop audio file", type=["wav","mp3","flac","ogg","m4a"],
                                  label_visibility="collapsed")
            if up:
                audio_bytes_input = up.getvalue()
                st.audio(audio_bytes_input)
                sz = len(audio_bytes_input) / 1024
                st.caption(f"📄 {up.name}  ·  {sz:.1f} KB")
        else:
            dur = st.slider("Recording duration (s)", 2, 10, RECORDING_DURATION, key="rec_dur")
            st.caption("⚠️ Ensure microphone is connected and browser permission granted.")
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                ph = st.empty()
                with ph.container():
                    st.info(f"🎙️ Recording for **{dur}s** … speak now!")
                    prog = st.progress(0)
                    for i in range(dur * 10):
                        time.sleep(0.1)
                        prog.progress((i + 1) / (dur * 10))
                try:
                    arr = sd.rec(int(dur * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                                 channels=1, dtype="float32")
                    sd.wait()
                    recorded_array = arr.flatten()
                    st.session_state.audio_array = recorded_array
                    buf = io.BytesIO()
                    sf.write(buf, recorded_array, SAMPLE_RATE, format="WAV")
                    audio_bytes_input = buf.getvalue()
                    ph.empty()
                    st.audio(audio_bytes_input, format="audio/wav")
                    st.success("✅ Recording complete!")
                except Exception as e:
                    ph.empty()
                    st.error(f"Recording failed: {e}")

        st.markdown("---")
        analyse = st.button("⚡ Analyse Emotion", type="primary",
                             use_container_width=True,
                             disabled=(audio_bytes_input is None and recorded_array is None))

    # ── Result panel ────────────────────────────────────────────────────
    with col_res:
        st.markdown("### 🎯 Result")
        predictor = get_predictor()

        if analyse and predictor:
            with st.spinner("Analysing audio …"):
                try:
                    if recorded_array is not None:
                        result   = predictor.predict_from_array(recorded_array, SAMPLE_RATE)
                        filename = "microphone_recording"
                    else:
                        result   = predictor.predict_from_bytes(audio_bytes_input, "uploaded_file")
                        filename = "uploaded_file"
                    st.session_state.last_result = result
                    st.session_state.audio_bytes = audio_bytes_input
                    st.session_state.audio_array = recorded_array
                    save_prediction(
                        emotion=result["emotion"], confidence=result["confidence"],
                        probabilities=result["probabilities"], model_name=result["model_name"],
                        filename=filename, duration_s=result["audio_info"].get("duration_s"),
                        session_id=st.session_state.session_id,
                        inference_ms=result.get("inference_time_ms"),
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

        result = st.session_state.last_result
        if result:
            emotion    = result["emotion"]
            confidence = result["confidence"]
            color      = result["color"]
            emoji      = result["emoji"]
            is_unc     = result.get("is_uncertain", False)

            # Uncertainty banner
            if is_unc:
                reason = result.get("uncertainty_reason", "Low confidence")
                st.warning(f"⚠️ **Low confidence** — {reason}. Results may be unreliable.")

            # Emotion badge + confidence
            st.markdown(f"""
            <div style="text-align:center;padding:18px 0 8px">
              <div class="emotion-badge" style="background:{color}20;border:2px solid {color};color:{color};margin:0 auto;width:fit-content">
                {emoji} {emotion.capitalize()}
              </div>
              <div class="confidence-num" style="margin-top:14px">{confidence*100:.1f}%</div>
              <div style="color:#64748b;font-size:.85rem;margin-top:2px;font-weight:500">
                Confidence · {result.get('model_version','v2')} · {result.get('inference_time_ms',0):.0f} ms
              </div>
            </div>""", unsafe_allow_html=True)

            # Top-3 predictions
            st.markdown('<div class="section-label" style="text-align:center;margin-top:4px">Top Predictions</div>', unsafe_allow_html=True)
            top3 = result.get("top_k", [])[:3]
            cols = st.columns(len(top3))
            for col, p in zip(cols, top3):
                col.metric(f"{p['emoji']} {p['emotion'].capitalize()}", p["pct"])

            # Probability chart
            st.plotly_chart(plot_emotion_probabilities(result["probabilities"], emotion),
                            use_container_width=True, key="det_probs")

            # Audio stats
            if result["audio_info"]:
                info = result["audio_info"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Duration", f"{info.get('duration_s',0):.2f}s")
                c2.metric("RMS", f"{info.get('rms',0):.4f}")
                c3.metric("Sample Rate", f"{info.get('sample_rate',0)} Hz")
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px 20px">
              <div style="font-size:3rem;opacity:.35">🎙️</div>
              <p style="color:#475569;font-size:.92rem;margin-top:10px">
                Upload or record audio,<br>then click <strong style="color:#818cf8">Analyse Emotion</strong>
              </p>
            </div>""", unsafe_allow_html=True)

        if not predictor:
            st.warning("⚠️ No model loaded. Run `python generate_demo_model.py`")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — BATCH PREDICT
# ═══════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📦 Batch Prediction")
    st.caption("Upload multiple audio files and get all predictions at once.")

    batch_files = st.file_uploader("Drop multiple audio files", type=["wav","mp3","flac","ogg"],
                                   accept_multiple_files=True, key="batch_upload")

    if batch_files and st.button("⚡ Analyse All Files", key="batch_btn", type="primary"):
        predictor = get_predictor()
        if not predictor:
            st.warning("No model loaded.")
        else:
            items   = [(f.getvalue(), f.name) for f in batch_files]
            results = []
            prog    = st.progress(0)
            status  = st.empty()
            for i, (ab, fn) in enumerate(items):
                status.caption(f"Processing {fn} …")
                try:
                    r = predictor.predict_from_bytes(ab, fn)
                    results.append((fn, r))
                    save_prediction(
                        emotion=r["emotion"], confidence=r["confidence"],
                        probabilities=r["probabilities"], model_name=r["model_name"],
                        filename=fn, duration_s=r["audio_info"].get("duration_s"),
                        session_id=st.session_state.session_id,
                        inference_ms=r.get("inference_time_ms"),
                    )
                except Exception as e:
                    results.append((fn, {"error": str(e)}))
                prog.progress((i + 1) / len(items))
            status.empty()

            st.success(f"✅ {len(results)} files processed")

            # Download CSV
            csv_rows = "filename,emotion,confidence,duration_s,inference_ms\n"
            for fn, r in results:
                if "error" not in r:
                    csv_rows += f"{fn},{r['emotion']},{r['confidence']*100:.1f},{r['audio_info'].get('duration_s',0):.2f},{r.get('inference_time_ms',0):.0f}\n"
                else:
                    csv_rows += f"{fn},ERROR,,,\n"
            st.download_button("⬇️ Download CSV Results", csv_rows,
                               file_name="batch_results.csv", mime="text/csv")

            # Results list
            for fn, r in results:
                if "error" in r:
                    st.error(f"❌ {fn}: {r['error']}")
                    continue
                color = r["color"]
                unc   = " ⚠️" if r.get("is_uncertain") else ""
                st.markdown(f"""
                <div class="history-row" style="--row-color:{color}">
                  <span style="font-size:1.4rem">{r['emoji']}</span>
                  <div style="flex:1">
                    <strong style="color:{color}">{r['emotion'].capitalize()}{unc}</strong>
                    <span style="color:#64748b;font-size:.82rem"> · {fn}</span>
                  </div>
                  <div style="text-align:right">
                    <span style="font-weight:700;color:#f1f5f9">{r['confidence']*100:.1f}%</span><br>
                    <span style="color:#475569;font-size:.75rem">{r['audio_info'].get('duration_s',0):.1f}s · {r.get('inference_time_ms',0):.0f}ms</span>
                  </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — AUDIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown("### 📊 Audio Signal Analysis")
    result = st.session_state.last_result

    if result is None:
        st.info("🔍 Analyse an audio file first (Detect Emotion tab) to see visualisations here.")
    else:
        # Explanation cards
        with st.expander("📖 What do these charts mean?", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown("""**🌊 Waveform**\nThe raw amplitude of the audio signal over time.
High peaks = loud sounds. Useful for spotting silence and energy patterns.""")
            c2.markdown("""**🎨 Mel Spectrogram**\nFrequency content over time on a human-perceptual scale.
Brighter = more energy. Key for detecting emotional pitch and rhythm.""")
            c3.markdown("""**📈 MFCC**\nMel-Frequency Cepstral Coefficients represent the voice timbre.
The model uses these as its primary input features.""")

        try:
            audio_bytes = st.session_state.get("audio_bytes")
            if audio_bytes:
                y, sr = preprocess_audio(audio_bytes=audio_bytes)
            else:
                y, sr = preprocess_audio(audio_array=st.session_state.get("audio_array"))

            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(plot_waveform(y, sr), use_container_width=True, key="viz_wave")
                st.plotly_chart(plot_mfcc(y, sr),     use_container_width=True, key="viz_mfcc")
            with v2:
                st.plotly_chart(plot_mel_spectrogram(y, sr), use_container_width=True, key="viz_mel")
                from utils.feature_extraction import extract_features
                feat = extract_features(y, sr)
                st.markdown("**📐 Feature Vector Statistics**")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Dimensions", len(feat))
                fc2.metric("Mean",       f"{feat.mean():.4f}")
                fc3.metric("Std Dev",    f"{feat.std():.4f}")
        except Exception as e:
            st.error(f"Visualisation error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — HISTORY  (searchable, filterable, paginated)
# ═══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Prediction History")

    # ── Filters ────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1, 1])
    with fc1:
        search_txt = st.text_input("🔍 Search", placeholder="emotion or filename …", label_visibility="collapsed")
    with fc2:
        emotions_in_db = ["All"] + get_distinct_emotions()
        emo_filter = st.selectbox("Emotion", emotions_in_db, label_visibility="collapsed")
    with fc3:
        conf_min = st.slider("Min conf %", 0, 100, 0, key="hist_conf_min")
    with fc4:
        page_size = st.selectbox("Per page", [10, 20, 50], index=1, label_visibility="collapsed")

    emo_arg  = None if emo_filter == "All" else emo_filter
    conf_min_f = conf_min / 100.0

    total = get_predictions_count(emotion_filter=emo_arg, min_confidence=conf_min_f, search_text=search_txt or None)
    n_pages = max(1, (total + page_size - 1) // page_size)
    page    = st.number_input("Page", 1, n_pages, 1, label_visibility="collapsed") if n_pages > 1 else 1
    offset  = (page - 1) * page_size

    history = get_predictions(limit=page_size, offset=offset, emotion_filter=emo_arg,
                              min_confidence=conf_min_f, search_text=search_txt or None)

    # Download filtered results
    csv_data = export_predictions_csv()
    if csv_data:
        hc1, hc2 = st.columns([3, 1])
        hc1.caption(f"Showing {len(history)} of {total} predictions (page {page}/{n_pages})")
        hc2.download_button("⬇️ Export CSV", csv_data, "history.csv", "text/csv")
    else:
        st.caption("No predictions yet.")

    h_col, trend_col = st.columns([2, 1])

    with h_col:
        if not history:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px">
              <div style="font-size:2.5rem;opacity:.3">📭</div>
              <p style="color:#475569;margin-top:8px">No predictions match your filters.</p>
            </div>""", unsafe_allow_html=True)
        else:
            for rec in history:
                color  = EMOTION_COLORS.get(rec["emotion"], "#94A3B8")
                emoji  = EMOTION_EMOJIS.get(rec["emotion"], "🎙️")
                ts_str = rec["timestamp"].replace("T", " ")
                fn     = rec.get("filename") or "—"
                conf   = rec["confidence"] * 100
                dur    = f"{rec.get('duration_s',0):.1f}s" if rec.get('duration_s') else "—"
                st.markdown(f"""
                <div class="history-row" style="--row-color:{color}">
                  <span style="font-size:1.35rem">{emoji}</span>
                  <div style="flex:1;min-width:0">
                    <strong style="color:{color}">{rec['emotion'].capitalize()}</strong>
                    <span style="color:#64748b;font-size:.8rem"> · {ts_str}</span><br>
                    <span style="color:#475569;font-size:.75rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block">{fn}</span>
                  </div>
                  <div style="text-align:right;white-space:nowrap">
                    <span style="font-weight:700;color:#f1f5f9">{conf:.1f}%</span><br>
                    <span style="color:#475569;font-size:.72rem">{dur}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

    with trend_col:
        st.markdown("#### Emotion Trend")
        trend = get_trend_data(n=30)
        st.plotly_chart(plot_emotion_trend(trend), use_container_width=True, key="hist_trend")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("### 🧠 Analytics Dashboard")
    stats = get_emotion_stats()

    if not stats:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px">
          <div style="font-size:3rem;opacity:.3">📊</div>
          <p style="color:#475569;margin-top:10px">No data yet — make some predictions first.</p>
        </div>""", unsafe_allow_html=True)
    else:
        total     = sum(v["count"] for v in stats.values())
        dom_emo   = max(stats, key=lambda k: stats[k]["count"])
        avg_conf  = np.mean([v["avg_confidence"] for v in stats.values()])
        max_conf  = max(v["max_confidence"] for v in stats.values())

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", total)
        k2.metric("Dominant Emotion",  f"{EMOTION_EMOJIS.get(dom_emo,'')} {dom_emo.capitalize()}")
        k3.metric("Avg Confidence",    f"{avg_conf*100:.1f}%")
        k4.metric("Peak Confidence",   f"{max_conf*100:.1f}%")

        st.divider()

        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(plot_emotion_distribution(stats),
                            use_container_width=True, key="dash_dist")
        with d2:
            comp_path = LOGS_DIR / "model_comparison.json"
            if comp_path.exists():
                comp = json.load(open(comp_path))
                st.plotly_chart(plot_model_comparison(comp),
                                use_container_width=True, key="dash_comp")
            else:
                import plotly.express as px
                emotions = list(stats.keys())
                counts   = [stats[e]["count"] for e in emotions]
                fig = px.bar(
                    x=[e.capitalize() for e in emotions], y=counts,
                    color=emotions,
                    color_discrete_map={e: EMOTION_COLORS.get(e,"#94A3B8") for e in emotions},
                    title="Prediction Count by Emotion", template="plotly_dark",
                )
                fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(255,255,255,.03)", font_color="#e2e8f0",
                                  height=320)
                st.plotly_chart(fig, use_container_width=True, key="dash_bar")

        # Trend + histogram
        trend_all = get_trend_data(n=100)
        d3, d4 = st.columns(2)
        with d3:
            st.plotly_chart(plot_emotion_trend(trend_all),
                            use_container_width=True, key="dash_trend")
        with d4:
            st.plotly_chart(plot_confidence_histogram(trend_all),
                            use_container_width=True, key="dash_hist")

        # Stats table
        st.markdown("#### 📋 Emotion Statistics")
        import pandas as pd
        df = pd.DataFrame([
            {"Emotion": e.capitalize(),
             "Count":   stats[e]["count"],
             "Avg Conf": f"{stats[e]['avg_confidence']*100:.1f}%",
             "Max Conf": f"{stats[e]['max_confidence']*100:.1f}%",
             "Share":    f"{stats[e]['count']/total*100:.1f}%"}
            for e in stats
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
