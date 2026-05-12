"""
visualizations.py
==================
Production-grade Plotly visualization helpers for VoiceEmo v2.

All charts:
- transparent backgrounds (match dark theme)
- readable axes, labels, tooltips
- consistent color palette
- proper margins and heights
- accessible text colors (#e2e8f0 / #c7d2fe)
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

# ── Shared layout defaults ─────────────────────────────────────────────────
_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=50, r=20, t=42, b=42),
)

def _layout(**kwargs) -> dict:
    """Merge chart-specific overrides into the base layout."""
    d = _BASE.copy()
    d.update(kwargs)
    return d


# ─────────────────────────────────────────────
#  Waveform
# ─────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE,
                  title: str = "Audio Waveform") -> go.Figure:
    """Interactive waveform with amplitude envelope shading."""
    times = np.linspace(0, len(y) / sr, num=len(y))

    fig = go.Figure()
    # Positive fill
    fig.add_trace(go.Scatter(
        x=times, y=np.where(y >= 0, y, 0),
        mode="lines", line=dict(color="#818cf8", width=1),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.15)",
        name="Amplitude (+)", showlegend=False,
    ))
    # Negative fill
    fig.add_trace(go.Scatter(
        x=times, y=np.where(y < 0, y, 0),
        mode="lines", line=dict(color="#c084fc", width=1),
        fill="tozeroy", fillcolor="rgba(192,132,252,0.12)",
        name="Amplitude (−)", showlegend=False,
    ))
    fig.update_layout(
        **_layout(
            title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Amplitude", gridcolor="rgba(255,255,255,0.06)"),
            height=220,
        )
    )
    return fig


# ─────────────────────────────────────────────
#  Mel Spectrogram
# ─────────────────────────────────────────────

def plot_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE,
                          title: str = "Mel Spectrogram") -> go.Figure:
    """Log-Mel spectrogram heatmap."""
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    times   = librosa.frames_to_time(np.arange(log_mel.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    freqs   = librosa.mel_frequencies(n_mels=N_MELS)

    fig = go.Figure(go.Heatmap(
        z=log_mel, x=times, y=freqs,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="dB", font=dict(color="#e2e8f0", size=12)),
            tickfont=dict(color="#e2e8f0", size=11),
        ),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
    ))
    fig.update_layout(
        **_layout(
            title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Frequency (Hz)", gridcolor="rgba(255,255,255,0.06)"),
            height=280,
        )
    )
    return fig


# ─────────────────────────────────────────────
#  MFCC Heatmap
# ─────────────────────────────────────────────

def plot_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE,
              n_mfcc: int = 20, title: str = "MFCC Coefficients") -> go.Figure:
    """MFCC heatmap with readable coefficient labels."""
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    fig = go.Figure(go.Heatmap(
        z=mfcc,
        x=times,
        y=[f"C{i+1}" for i in range(n_mfcc)],
        colorscale="RdBu",
        colorbar=dict(
            title=dict(text="Value", font=dict(color="#e2e8f0", size=12)),
            tickfont=dict(color="#e2e8f0", size=11),
        ),
        hovertemplate="Time: %{x:.2f}s<br>%{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_layout(
            title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Coefficient", tickfont=dict(size=10),
                       gridcolor="rgba(255,255,255,0.06)"),
            height=280, margin=dict(l=55, r=20, t=42, b=42),
        )
    )
    return fig


# ─────────────────────────────────────────────
#  Emotion Probability Bar Chart
# ─────────────────────────────────────────────

def plot_emotion_probabilities(probabilities: dict,
                                predicted_emotion: str) -> go.Figure:
    """
    Horizontal bar chart of all emotion probabilities.
    Predicted emotion is highlighted; others are muted.
    Includes percentage labels and hover detail.
    """
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1])
    emotions     = [e for e, _ in sorted_items]
    probs        = [p * 100 for _, p in sorted_items]
    colors       = [
        EMOTION_COLORS.get(e, "#94A3B8") if e == predicted_emotion
        else "rgba(255,255,255,0.12)"
        for e in emotions
    ]

    fig = go.Figure(go.Bar(
        y=[e.capitalize() for e in emotions],
        x=probs,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_layout(
            title=dict(text="Probability Distribution", font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Confidence (%)", range=[0, 120],
                       gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(size=11)),
            height=max(240, len(emotions) * 46),
            margin=dict(l=20, r=70, t=42, b=42),
            showlegend=False,
        )
    )
    return fig


# ─────────────────────────────────────────────
#  Emotion Trend Over Time
# ─────────────────────────────────────────────

def plot_emotion_trend(trend_data: list[dict]) -> go.Figure:
    """
    Line chart of emotion confidence over time.
    Each unique emotion is a separate coloured trace.
    """
    if not trend_data:
        fig = go.Figure()
        fig.update_layout(
            **_layout(
                title=dict(text="No trend data yet — make some predictions!",
                           font=dict(size=13, color="#64748b")),
                height=280,
            )
        )
        return fig

    df = pd.DataFrame(trend_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["conf_pct"]  = df["confidence"] * 100

    fig = go.Figure()
    for emotion in df["emotion"].unique():
        sub = df[df["emotion"] == emotion].sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=sub["timestamp"],
            y=sub["conf_pct"],
            mode="lines+markers",
            name=emotion.capitalize(),
            line=dict(color=EMOTION_COLORS.get(emotion, "#94A3B8"), width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{x|%H:%M:%S}<br>" + emotion.capitalize() + ": %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_layout(
            title=dict(text="Confidence Trend Over Time", font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Time", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Confidence (%)", range=[0, 105],
                       gridcolor="rgba(255,255,255,0.06)"),
            height=300,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11, color="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    return fig


# ─────────────────────────────────────────────
#  Emotion Distribution Donut
# ─────────────────────────────────────────────

def plot_emotion_distribution(stats: dict) -> go.Figure:
    """
    Donut chart of emotion counts with hover tooltips and pull effect on
    the dominant emotion.
    """
    if not stats:
        fig = go.Figure()
        fig.update_layout(
            **_layout(
                title=dict(text="No data yet", font=dict(size=13, color="#64748b")),
                height=300,
            )
        )
        return fig

    emotions = list(stats.keys())
    counts   = [stats[e]["count"] for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#94A3B8") for e in emotions]
    dom_idx  = counts.index(max(counts))
    pull     = [0.08 if i == dom_idx else 0 for i in range(len(emotions))]

    fig = go.Figure(go.Pie(
        labels=[e.capitalize() for e in emotions],
        values=counts,
        hole=0.5,
        pull=pull,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.4)", width=2)),
        textfont=dict(color="#ffffff", size=12),
        hovertemplate="%{label}: %{value} predictions (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_layout(
            title=dict(text="Emotion Distribution", font=dict(size=14, color="#e2e8f0")),
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                font=dict(size=11, color="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    return fig


# ─────────────────────────────────────────────
#  Confidence Distribution Histogram
# ─────────────────────────────────────────────

def plot_confidence_histogram(trend_data: list[dict]) -> go.Figure:
    """
    Histogram of confidence scores across all predictions.
    Helps users understand the model's certainty profile.
    """
    if not trend_data:
        fig = go.Figure()
        fig.update_layout(**_layout(height=260))
        return fig

    confs = [d["confidence"] * 100 for d in trend_data]
    fig   = go.Figure(go.Histogram(
        x=confs,
        nbinsx=20,
        marker=dict(
            color="rgba(129,140,248,0.7)",
            line=dict(color="rgba(129,140,248,1)", width=1),
        ),
        hovertemplate="Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>",
        name="Predictions",
    ))
    fig.update_layout(
        **_layout(
            title=dict(text="Confidence Score Distribution", font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Confidence (%)", range=[0, 105],
                       gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Predictions", gridcolor="rgba(255,255,255,0.06)"),
            height=260,
            showlegend=False,
        )
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
    results : {model_name: {"accuracy": float, "f1_weighted": float}}
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
        hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="F1 Score", x=models, y=f1,
        marker_color="#A78BFA",
        text=[f"{v:.1f}%" for v in f1],
        textposition="outside",
        hovertemplate="%{x}<br>F1: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_layout(
            title=dict(text="Model Performance Comparison",
                       font=dict(size=14, color="#e2e8f0")),
            yaxis=dict(title="Score (%)", range=[0, 115],
                       gridcolor="rgba(255,255,255,0.06)"),
            barmode="group",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
    )
    return fig


# ─────────────────────────────────────────────
#  Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray,
                           labels: list[str],
                           title: str = "Confusion Matrix") -> go.Figure:
    """Annotated heatmap confusion matrix."""
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[l.capitalize() for l in labels],
        y=[l.capitalize() for l in labels],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(color="white", size=11),
        colorbar=dict(
            title=dict(text="Count", font=dict(color="#e2e8f0", size=12)),
            tickfont=dict(color="#e2e8f0", size=11),
        ),
    ))
    fig.update_layout(
        **_layout(
            title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="Actual"),
            height=420,
            margin=dict(l=80, r=20, t=55, b=80),
        )
    )
    return fig
