"""
visualizations.py
==================
Plotly-based visualization helpers consumed by the Streamlit app.
All functions return a plotly Figure object.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import librosa
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import EMOTION_COLORS, SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS


# ─────────────────────────────────────────────
#  Waveform
# ─────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE,
                  title: str = "Audio Waveform") -> go.Figure:
    """Plotly interactive waveform plot."""
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=y,
        mode="lines",
        line=dict(color="#60A5FA", width=1),
        name="Amplitude"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=220,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
    )
    return fig


# ─────────────────────────────────────────────
#  Mel Spectrogram
# ─────────────────────────────────────────────

def plot_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE,
                          title: str = "Mel Spectrogram") -> go.Figure:
    """Plotly heatmap of the log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    times = librosa.frames_to_time(
        np.arange(log_mel.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )
    freqs = librosa.mel_frequencies(n_mels=N_MELS)

    fig = go.Figure(go.Heatmap(
        z=log_mel,
        x=times,
        y=freqs,
        colorscale="Viridis",
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_dark",
        height=280,
        margin=dict(l=60, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
    )
    return fig


# ─────────────────────────────────────────────
#  MFCC Heatmap
# ─────────────────────────────────────────────

def plot_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE,
              n_mfcc: int = 20, title: str = "MFCC") -> go.Figure:
    """Heatmap of MFCC coefficients over time."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    times = librosa.frames_to_time(
        np.arange(mfcc.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )
    fig = go.Figure(go.Heatmap(
        z=mfcc,
        x=times,
        y=[f"MFCC {i+1}" for i in range(n_mfcc)],
        colorscale="RdBu",
        colorbar=dict(title="Coeff"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Coefficient",
        template="plotly_dark",
        height=280,
        margin=dict(l=80, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
    )
    return fig


# ─────────────────────────────────────────────
#  Emotion Probability Bar Chart
# ─────────────────────────────────────────────

def plot_emotion_probabilities(probabilities: dict,
                                predicted_emotion: str) -> go.Figure:
    """
    Horizontal bar chart of emotion probabilities.
    The predicted emotion bar is highlighted.
    """
    emotions = list(probabilities.keys())
    probs    = [probabilities[e] * 100 for e in emotions]
    colors   = [
        EMOTION_COLORS.get(e, "#94A3B8") if e == predicted_emotion
        else "#334155"
        for e in emotions
    ]

    fig = go.Figure(go.Bar(
        y=emotions,
        x=probs,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 115]),
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
#  Emotion Trend Over Time
# ─────────────────────────────────────────────

def plot_emotion_trend(trend_data: list[dict]) -> go.Figure:
    """
    Line chart showing predicted emotion confidence over time.
    Each emotion is a separate trace.
    """
    if not trend_data:
        fig = go.Figure()
        fig.update_layout(title="No trend data yet", template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig

    df = pd.DataFrame(trend_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    for emotion in df["emotion"].unique():
        sub = df[df["emotion"] == emotion]
        fig.add_trace(go.Scatter(
            x=sub["timestamp"],
            y=sub["confidence"] * 100,
            mode="lines+markers",
            name=emotion.capitalize(),
            line=dict(color=EMOTION_COLORS.get(emotion, "#94A3B8"), width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="Emotion Trend Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ─────────────────────────────────────────────
#  Session Emotion Pie
# ─────────────────────────────────────────────

def plot_emotion_distribution(stats: dict) -> go.Figure:
    """
    Pie / donut chart of emotion counts across the session.
    """
    if not stats:
        fig = go.Figure()
        fig.update_layout(title="No session data", template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig

    emotions = list(stats.keys())
    counts   = [stats[e]["count"] for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#94A3B8") for e in emotions]

    fig = go.Figure(go.Pie(
        labels=[e.capitalize() for e in emotions],
        values=counts,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title="Session Emotion Distribution",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────
#  Confusion Matrix Heatmap
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray,
                           labels: list[str],
                           title: str = "Confusion Matrix") -> go.Figure:
    """Plotly heatmap for sklearn confusion matrix."""
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[l.capitalize() for l in labels],
        y=[l.capitalize() for l in labels],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(color="white"),
        colorbar=dict(title="Count"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        height=420,
        margin=dict(l=80, r=20, t=50, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
    )
    return fig


# ─────────────────────────────────────────────
#  Model Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_model_comparison(results: dict) -> go.Figure:
    """
    Grouped bar chart comparing accuracy and F1 across models.

    Parameters
    ----------
    results : dict
        {model_name: {"accuracy": float, "f1": float, ...}}
    """
    models   = list(results.keys())
    accuracy = [results[m].get("accuracy", 0) * 100 for m in models]
    f1       = [results[m].get("f1_weighted", 0) * 100 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=models, y=accuracy,
        marker_color="#60A5FA",
        text=[f"{v:.1f}%" for v in accuracy],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="F1 Score", x=models, y=f1,
        marker_color="#A78BFA",
        text=[f"{v:.1f}%" for v in f1],
        textposition="outside",
    ))
    fig.update_layout(
        title="Model Performance Comparison",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 115]),
        barmode="group",
        template="plotly_dark",
        height=360,
        margin=dict(l=40, r=20, t=50, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
