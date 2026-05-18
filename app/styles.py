"""styles.py — VoiceEmo v2 design system injected via st.markdown."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── TOKENS ─────────────────────────────────────────── */
:root {
  --bg:        #0a0d1a;
  --surface:   #111425;
  --card:      #161b30;
  --border:    rgba(129,140,248,.14);
  --border-hi: rgba(129,140,248,.28);
  --accent:    #818cf8;
  --accent2:   #c084fc;
  --txt-hi:    #f1f5f9;
  --txt-md:    #cbd5e1;
  --txt-lo:    #94a3b8;
  --txt-dim:   #64748b;
  --success:   #34d399;
  --warn:      #f59e0b;
  --danger:    #f87171;
}

/* ── BASE ────────────────────────────────────────────── */
html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--txt-md); }
#MainMenu, header, footer   { visibility:hidden; }
[data-testid="stHeader"]    { background:transparent; }
.block-container            { padding-top:1.2rem !important; max-width:1280px; }
.stApp { background: linear-gradient(160deg,#09091a 0%,#0d1028 40%,#10122a 100%); }

/* ── SIDEBAR ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0c0f22 0%,#0e1127 100%) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 { color:#fff !important; font-size:1.15rem !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span { color:var(--txt-md) !important; }
[data-testid="stSidebar"] .stMarkdown strong { color:var(--txt-hi) !important; }
[data-testid="stSidebar"] hr { border-color:var(--border) !important; margin:10px 0 !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#c7d2fe !important; font-size:1.7rem !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color:var(--txt-lo) !important; font-size:.78rem !important; text-transform:uppercase; letter-spacing:.5px; }

/* ── METRICS ─────────────────────────────────────────── */
div[data-testid="metric-container"] {
  background: rgba(129,140,248,.06);
  border: 1px solid var(--border);
  border-radius:14px; padding:16px 20px;
  box-shadow:0 2px 12px rgba(0,0,0,.2);
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
  color:var(--accent) !important; font-weight:600 !important;
  font-size:.78rem !important; text-transform:uppercase; letter-spacing:.5px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color:var(--txt-hi) !important; font-weight:700 !important;
}

/* ── TABS ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background:rgba(129,140,248,.05);
  border:1px solid var(--border); border-radius:14px; padding:4px; gap:2px;
}
.stTabs [data-baseweb="tab"] {
  border-radius:10px; font-weight:500; font-size:.88rem !important;
  color:var(--txt-lo) !important; padding:8px 16px !important; transition:all .2s;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,rgba(99,102,241,.4),rgba(139,92,246,.25)) !important;
  color:#fff !important; font-weight:700;
  box-shadow:0 2px 10px rgba(99,102,241,.25);
}
.stTabs [data-baseweb="tab"]:hover { color:#e0e7ff !important; background:rgba(99,102,241,.12) !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ── BUTTONS ─────────────────────────────────────────── */
.stButton > button {
  border-radius:12px !important; font-weight:600 !important;
  font-size:.93rem !important; padding:.55rem 1.4rem !important;
  transition:all .22s cubic-bezier(.4,0,.2,1) !important;
  color:#fff !important; letter-spacing:.2px;
}
.stButton > button[kind="primary"], .stButton > button:not([kind]) {
  background: linear-gradient(135deg,#6366f1,#8b5cf6 60%,#a78bfa) !important;
  border:none !important; box-shadow:0 3px 14px rgba(99,102,241,.35);
}
.stButton > button[kind="secondary"] {
  background:rgba(255,255,255,.06) !important;
  border:1px solid var(--border-hi) !important; color:#c7d2fe !important;
}
.stButton > button:hover {
  transform:translateY(-2px);
  box-shadow:0 8px 28px rgba(99,102,241,.45) !important; filter:brightness(1.08);
}
.stButton > button:active  { transform:translateY(0) !important; }
.stButton > button:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 3px !important;
  box-shadow: 0 0 0 4px rgba(129,140,248,.2) !important;
}
.stButton > button:disabled { opacity:.45 !important; transform:none !important; }

/* ── UPLOAD ──────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background:rgba(99,102,241,.04); border:2px dashed rgba(129,140,248,.35);
  border-radius:16px; padding:8px !important; transition:all .2s;
}
[data-testid="stFileUploader"]:hover {
  border-color:rgba(129,140,248,.6); background:rgba(99,102,241,.07);
}
[data-testid="stFileUploader"] section > div > span { color:#c7d2fe !important; }
[data-testid="stFileUploader"] small { color:var(--txt-lo) !important; }

/* ── FORM CONTROLS ───────────────────────────────────── */
.stRadio > div { color:var(--txt-hi) !important; }
.stRadio label span { color:var(--txt-hi) !important; font-weight:500; font-size:.92rem; }
.stRadio label:hover span { color:#fff !important; }
.stSlider label { color:var(--accent) !important; font-weight:600 !important; font-size:.85rem !important; }
.stSelectbox label { color:var(--accent) !important; font-weight:600 !important; font-size:.85rem !important; }
.stSelectbox > div > div { color:var(--txt-hi) !important; }
.stTextInput label { color:var(--accent) !important; font-weight:600 !important; }
.stTextInput input { background:rgba(255,255,255,.05) !important; color:var(--txt-hi) !important; border-color:var(--border) !important; border-radius:10px !important; }
.stTextInput input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 2px rgba(129,140,248,.2) !important; }

/* ── CARDS ───────────────────────────────────────────── */
.ve-card {
  background:rgba(255,255,255,.035); border:1px solid var(--border);
  border-radius:16px; padding:24px;
  box-shadow:0 4px 24px rgba(0,0,0,.25); margin-bottom:16px;
}
.ve-card-sm {
  background:rgba(255,255,255,.03); border:1px solid var(--border);
  border-radius:12px; padding:16px; margin-bottom:10px;
}

/* ── HISTORY ROW ─────────────────────────────────────── */
.history-row {
  display:flex; align-items:center; gap:14px;
  padding:14px 20px; border-radius:12px;
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.06);
  border-left:4px solid var(--row-color,#818cf8);
  margin-bottom:10px; transition:background .2s;
}
.history-row:hover { background:rgba(255,255,255,.06); }

/* ── TYPOGRAPHY ──────────────────────────────────────── */
.stMarkdown h1 { color:var(--txt-hi) !important; }
.stMarkdown h2 { color:var(--txt-hi) !important; }
.stMarkdown h3 { color:var(--txt-hi) !important; font-weight:700 !important; font-size:1.25rem !important; }
.stMarkdown h4 { color:var(--txt-md) !important; font-weight:600 !important; font-size:1.05rem !important; }
.stMarkdown p  { color:var(--txt-md) !important; line-height:1.65 !important; }
.stCaption p   { color:var(--txt-lo) !important; font-size:.82rem !important; }
hr { border-color:var(--border) !important; }
[data-testid="stAlert"] { border-radius:12px !important; }

/* ── CHARTS ──────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
  background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06);
  border-radius:14px; padding:4px;
}

/* ── DATAFRAME ───────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid var(--border); }

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(129,140,248,.22); border-radius:3px; }

/* ── BADGES ──────────────────────────────────────────── */
.emotion-badge {
  display:inline-flex; align-items:center; gap:12px;
  padding:14px 32px; border-radius:50px;
  font-size:1.45rem; font-weight:700;
  animation:badge-glow 2.5s ease-in-out infinite;
}
@keyframes badge-glow {
  0%,100% { box-shadow:0 0 0 0 rgba(99,102,241,.3); }
  50%     { box-shadow:0 0 0 12px rgba(99,102,241,0); }
}
.confidence-num {
  font-size:3rem; font-weight:800; line-height:1.1;
  background:linear-gradient(135deg,#818cf8,#c084fc);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.section-label {
  font-size:.75rem; font-weight:700; letter-spacing:.8px;
  text-transform:uppercase; color:var(--txt-lo); margin-bottom:6px;
}
audio { border-radius:10px; width:100%; }
</style>
"""
