"""
MULTIMODAL SENTIMENT DASHBOARD — v4.1 (Hume AI — Docs-Accurate)
════════════════════════════════════════════════════════════════
Corrected using official Hume docs:
  • Streaming: Config(face={}) / Config(language=StreamLanguage(...))
  • send_file(path, config=...) → result.face.predictions[0].emotions
  • emotion.name (Title Case) → lowercased for storage
  • Language model: 53 emotions + sentiment (9-pt) + toxicity (6 cats)

Tabs:
  📷 Face     — webcam → Hume face streaming → 48 emotions + FACS + descriptions
  💬 Language — text input → Hume language streaming → 53 emotions + sentiment + toxicity
  📊 Analytics — cross-session charts: valence timeline, radar, heatmap, arousal scatter

Install:
  pip install streamlit hume pillow plotly scikit-learn

Run:
  streamlit run streamlit_sentiment_hume_v4.py
"""

import streamlit as st
import asyncio
import io
import os
import tempfile
import numpy as np
import time
import random
from collections import Counter
from PIL import Image

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="⬡ HUME SENTIMENT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Graceful imports ──────────────────────────────────────────
try:
    from hume import AsyncHumeClient
    from hume.expression_measurement.stream import Config
    try:
        from hume.expression_measurement.stream import StreamLanguage
    except ImportError:
        StreamLanguage = None   # older SDK — pass language as dict
    HUME_OK = True
except ImportError:
    HUME_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    SK_OK = True
except ImportError:
    SK_OK = False

# ══════════════════════════════════════════════════════════════
#  CONSTANTS  (verified against Hume official docs)
# ══════════════════════════════════════════════════════════════

# 48 face emotions (stored lowercase; Hume returns Title Case)
FACE_48 = [
    "admiration", "adoration", "aesthetic appreciation", "amusement", "anger",
    "anxiety", "awe", "awkwardness", "boredom", "calmness", "concentration",
    "contemplation", "confusion", "contempt", "contentment", "craving", "desire",
    "determination", "disappointment", "disgust", "distress", "doubt", "ecstasy",
    "embarrassment", "empathic pain", "entrancement", "envy", "excitement",
    "fear", "guilt", "horror", "interest", "joy", "love", "nostalgia", "pain",
    "pride", "realization", "relief", "romance", "sadness", "satisfaction",
    "shame", "surprise (negative)", "surprise (positive)", "sympathy",
    "tiredness", "triumph",
]

# 53 language emotions = 48 + 5 language-exclusive
LANG_EXTRAS = ["annoyance", "disapproval", "enthusiasm", "gratitude", "sarcasm"]
LANG_53 = FACE_48 + LANG_EXTRAS

# Toxicity categories
TOXICITY_CATS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# FACS action units supported by Hume
FACS_UNITS = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU16", "AU17", "AU18",
    "AU22", "AU23", "AU24", "AU25", "AU26", "AU27",
]
FACS_NAMES = {
    "AU1": "Inner Brow Raise",  "AU2": "Outer Brow Raise",  "AU4": "Brow Lowerer",
    "AU5": "Upper Lid Raise",   "AU6": "Cheek Raise",       "AU7": "Lids Tight",
    "AU9": "Nose Wrinkle",      "AU10": "Upper Lip Raiser", "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",          "AU15": "Lip Corner Depress","AU25": "Lips Part",
    "AU26": "Jaw Drop",         "AU27": "Mouth Stretch",
}

# Valence groups
POSITIVE = {
    "admiration", "adoration", "aesthetic appreciation", "amusement", "awe",
    "calmness", "contentment", "curiosity", "desire", "determination", "ecstasy",
    "enthusiasm", "entrancement", "excitement", "gratitude", "interest", "joy",
    "love", "pride", "relief", "romance", "satisfaction", "surprise (positive)",
    "triumph",
}
NEGATIVE = {
    "anger", "annoyance", "anxiety", "awkwardness", "boredom", "contempt",
    "craving", "disappointment", "disapproval", "disgust", "distress", "doubt",
    "embarrassment", "empathic pain", "envy", "fear", "guilt", "horror",
    "nostalgia", "pain", "sadness", "sarcasm", "shame", "surprise (negative)",
    "sympathy", "tiredness",
}
HIGH_AROUSAL = {
    "anger", "excitement", "ecstasy", "fear", "horror", "triumph", "enthusiasm",
    "amusement", "surprise (positive)", "surprise (negative)", "distress", "awe",
}
LOW_AROUSAL = {
    "calmness", "tiredness", "boredom", "contentment", "satisfaction",
    "nostalgia", "sadness", "disappointment",
}

# ── Palette ───────────────────────────────────────────────────
BG, CARD = "#050a0e", "#0d1620"
ACCENT   = "#00e5ff"
POS_C    = "#2DFF7A"
NEG_C    = "#FF3B3B"
NEU_C    = "#9A9A9A"
WARN_C   = "#FFB800"

def emo_color(name: str) -> str:
    if name in POSITIVE: return POS_C
    if name in NEGATIVE:  return NEG_C
    return NEU_C

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  html, body, [class*="css"] {{ font-family:'Share Tech Mono',monospace !important; }}
  .main .block-container {{ padding-top:0.7rem; }}
  h1,h2,h3,h4 {{ color:{ACCENT} !important; }}
  [data-testid="metric-container"] {{
    background:{CARD}; border:1px solid rgba(0,229,255,0.2);
    border-radius:6px; padding:8px 14px;
  }}
  .tag {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:.75rem; }}
  .emo-row {{ display:flex; align-items:center; gap:8px; margin:2px 0; }}
  .dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
_defaults = {
    "face_frames":  [],      # [{emotions, facs, descriptions, ts}]
    "last_face":    None,
    "prev_img_id":  None,
    "lang_results": [],      # [{emotions, sentiment, toxicity, text, ts}]
    "last_lang":    None,
    "api_key":      "",
    "running":      False,
    "enable_facs":  False,
    "enable_desc":  False,
    "ml_pipe":      None,
    "ml_acc":       0.0,
    "ml_trained":   False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
ss = st.session_state

MAX_HIST  = 300
MIN_TRAIN = 20

# ══════════════════════════════════════════════════════════════
#  HUME API  — FACE (streaming)
# ══════════════════════════════════════════════════════════════
async def _hume_face_async(api_key: str, img_bytes: bytes,
                            facs: bool, desc: bool) -> dict:
    """
    Hume streaming face analysis.
    Docs pattern:
        async with client.expression_measurement.stream.connect() as socket:
            result = await socket.send_file(path, config=Config(face={}))
        face_preds = result.face.predictions
        emotion.name   → Title Case string
        emotion.score  → float
    """
    client = AsyncHumeClient(api_key=api_key)

    # Build face config dict (extra outputs optional)
    face_cfg: dict = {}
    if facs: face_cfg["facs"] = {}
    if desc: face_cfg["descriptions"] = {}

    # Write to temp file — streaming SDK expects a file path
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.write(fd, img_bytes)
        os.close(fd)

        async with client.expression_measurement.stream.connect() as socket:
            result = await socket.send_file(tmp_path, config=Config(face=face_cfg))
    finally:
        try: os.unlink(tmp_path)
        except: pass

    out = {"emotions": {}, "facs": {}, "descriptions": {}}
    preds = getattr(result.face, "predictions", []) if result.face else []
    if preds:
        p0 = preds[0]
        for e in (getattr(p0, "emotions", []) or []):
            out["emotions"][e.name.lower()] = float(e.score)
        if facs:
            for f in (getattr(p0, "facs", []) or []):
                out["facs"][f.name] = float(f.score)
        if desc:
            for d in (getattr(p0, "descriptions", []) or []):
                out["descriptions"][d.name] = float(d.score)
    return out


def call_hume_face(api_key: str, pil_img: Image.Image,
                   facs: bool = False, desc: bool = False) -> dict | None:
    pil_img.thumbnail((640, 480), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    try:
        loop = asyncio.new_event_loop()
        r = loop.run_until_complete(_hume_face_async(api_key, buf.getvalue(), facs, desc))
        loop.close()
        return r
    except Exception as e:
        st.error(f"Hume face error: {e}")
        return None

# ══════════════════════════════════════════════════════════════
#  HUME API  — LANGUAGE (streaming)
# ══════════════════════════════════════════════════════════════
async def _hume_lang_async(api_key: str, text: str) -> dict:
    """
    Hume streaming language analysis.
    Docs pattern:
        result = await socket.send_text(
            text="...",
            config=Config(language=StreamLanguage(granularity="sentence"))
        )
        lang_preds = result.language.predictions
        prediction.emotions / .sentiment / .toxicity
    """
    client = AsyncHumeClient(api_key=api_key)

    if StreamLanguage is not None:
        lang_cfg = StreamLanguage(granularity="passage", sentiment={}, toxicity={})
        cfg = Config(language=lang_cfg)
    else:
        cfg = Config(language={"granularity": "passage", "sentiment": {}, "toxicity": {}})  # type: ignore

    async with client.expression_measurement.stream.connect() as socket:
        result = await socket.send_text(text=text, config=cfg)

    out = {"emotions": {}, "sentiment": [0.0] * 9, "toxicity": {}}
    preds = getattr(result.language, "predictions", []) if result.language else []
    if preds:
        p0 = preds[0]
        for e in (getattr(p0, "emotions", []) or []):
            out["emotions"][e.name.lower()] = float(e.score)
        for s in (getattr(p0, "sentiment", []) or []):
            idx = int(s.name) - 1   # "1"→0 … "9"→8
            if 0 <= idx < 9:
                out["sentiment"][idx] = float(s.score)
        for t in (getattr(p0, "toxicity", []) or []):
            out["toxicity"][t.name] = float(t.score)
    return out


def call_hume_language(api_key: str, text: str) -> dict | None:
    try:
        loop = asyncio.new_event_loop()
        r = loop.run_until_complete(_hume_lang_async(api_key, text))
        loop.close()
        return r
    except Exception as e:
        st.error(f"Hume language error: {e}")
        return None

# ══════════════════════════════════════════════════════════════
#  MOCK DATA
# ══════════════════════════════════════════════════════════════
_mock_state = {"top": ["joy", "calmness"]}

def _mock_emos(emo_list: list) -> dict:
    d = {e: random.uniform(0.005, 0.08) for e in emo_list}
    if random.random() < 0.3:
        _mock_state["top"] = random.choices(emo_list, k=random.randint(1, 3))
    for t in _mock_state["top"]:
        d[t] = random.uniform(0.4, 0.82)
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

def mock_face() -> dict:
    return {
        "emotions":     _mock_emos(FACE_48),
        "facs":         {au: random.uniform(0, 0.9) for au in FACS_UNITS[:10]},
        "descriptions": {"Smile": 0.72, "Grin": 0.45, "Wide-eyed": 0.31},
    }

def mock_language(text: str) -> dict:
    return {
        "emotions":  _mock_emos(LANG_53),
        "sentiment": list(np.random.dirichlet(np.ones(9))),
        "toxicity":  {c: random.uniform(0, 0.03) for c in TOXICITY_CATS},
    }

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def valence(emos: dict) -> tuple:
    p = sum(v for k, v in emos.items() if k in POSITIVE)
    n = sum(v for k, v in emos.items() if k in NEGATIVE)
    return p, n, max(0.0, 1.0 - p - n)

def arousal(emos: dict) -> float:
    hi = sum(v for k, v in emos.items() if k in HIGH_AROUSAL)
    lo = sum(v for k, v in emos.items() if k in LOW_AROUSAL)
    return (hi - lo + 1) / 2

def top_n(emos: dict, n: int = 10):
    return sorted(emos.items(), key=lambda x: -x[1])[:n]

def sentiment_score(dist: list) -> float:
    return sum((i + 1) * s for i, s in enumerate(dist))

# ══════════════════════════════════════════════════════════════
#  ML
# ══════════════════════════════════════════════════════════════
def _build_pipe():
    if not SK_OK: return None
    return Pipeline([("sc", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=80, random_state=42))])

if ss.ml_pipe is None:
    ss.ml_pipe = _build_pipe()

def train_ml():
    if not SK_OK or len(ss.face_frames) < MIN_TRAIN: return
    X, y = [], []
    for fr in ss.face_frames:
        vec = [fr["emotions"].get(e, 0) for e in FACE_48]
        p, n, _ = valence(fr["emotions"])
        y.append(0 if p >= n else 1)
        X.append(vec)
    if len(set(y)) < 2: return
    ss.ml_pipe.fit(np.array(X), np.array(y))
    ss.ml_trained = True
    ss.ml_acc = round(accuracy_score(y, ss.ml_pipe.predict(np.array(X))) * 100, 1)

# ══════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════
_L = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color="#8fa0b0", family="'Share Tech Mono',monospace", size=10),
    margin=dict(l=40, r=20, t=40, b=30),
    showlegend=False,
)

def _empty(title, h=280):
    f = go.Figure(layout=go.Layout(**{**_L, "title": title, "height": h}))
    f.add_annotation(text="Waiting for data…", xref="paper", yref="paper",
                     x=0.5, y=0.5, showarrow=False, font=dict(color="#334", size=13))
    return f

def fig_bars(emos: dict, title: str = "🎭 TOP EMOTIONS", n: int = 15) -> go.Figure:
    items = top_n(emos, n)[::-1]
    names  = [i[0] for i in items]
    scores = [i[1] * 100 for i in items]
    colors = [emo_color(n_) for n_ in names]
    f = go.Figure(layout=go.Layout(**{**_L, "title": title, "height": 400,
        "xaxis": dict(title="Score %", gridcolor="rgba(255,255,255,0.05)"),
        "yaxis": dict(showgrid=False)}))
    f.add_trace(go.Bar(x=scores, y=names, orientation="h",
        marker=dict(color=colors, opacity=0.88),
        text=[f"{s:.1f}%" for s in scores], textposition="outside"))
    return f

def fig_radar(emos: dict) -> go.Figure:
    cats = ["joy", "excitement", "anger", "fear", "sadness", "disgust",
            "awe", "calmness", "curiosity", "amusement", "determination", "anxiety"]
    vals = [emos.get(c, 0) * 100 for c in cats]
    cv, cc = vals + [vals[0]], cats + [cats[0]]
    f = go.Figure(layout=go.Layout(**{**_L, "title": "🕸️ EMOTION RADAR", "height": 400,
        "polar": dict(bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 70],
                            gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"))}))
    f.add_trace(go.Scatterpolar(r=cv, theta=cc, fill="toself",
        fillcolor="rgba(0,229,255,0.12)", line=dict(color=ACCENT, width=2),
        mode="lines+markers", marker=dict(color=ACCENT, size=5)))
    return f

def fig_valence_timeline(frames: list) -> go.Figure:
    if len(frames) < 2: return _empty("📊 VALENCE OVER TIME")
    recent = frames[-100:]
    ps, ns, us = zip(*[valence(fr["emotions"]) for fr in recent])
    x = list(range(len(ps)))
    f = go.Figure(layout=go.Layout(**{**_L, "title": "📊 VALENCE OVER TIME", "height": 280,
        "showlegend": True,
        "legend": dict(orientation="h", y=1.18, font=dict(size=9)),
        "yaxis": dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1.15]),
        "xaxis": dict(showgrid=False)}))
    for vals, name, color, fill in [
        (list(ps), "Positive", POS_C, "rgba(45,255,122,0.12)"),
        (list(ns), "Negative", NEG_C, "rgba(255,59,59,0.10)"),
        (list(us), "Neutral",  NEU_C, "rgba(154,154,154,0.07)"),
    ]:
        f.add_trace(go.Scatter(x=x, y=vals, name=name, fill="tozeroy", mode="lines",
                               line=dict(color=color, width=1.5), fillcolor=fill))
    return f

def fig_arousal_scatter(frames: list) -> go.Figure:
    if len(frames) < 3: return _empty("⚡ VALENCE × AROUSAL")
    recent = frames[-120:]
    xs, ys, labels, clrs = [], [], [], []
    for fr in recent:
        p, n, _ = valence(fr["emotions"])
        xs.append((p - n + 1) / 2)
        ys.append(arousal(fr["emotions"]))
        te = max(fr["emotions"], key=fr["emotions"].get)
        labels.append(te); clrs.append(emo_color(te))
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "⚡ VALENCE × AROUSAL (Russell Circumplex)", "height": 320,
        "xaxis": dict(title="← Negative | Positive →", range=[0,1],
                      gridcolor="rgba(255,255,255,0.05)"),
        "yaxis": dict(title="← Calm | Activated →", range=[0,1],
                      gridcolor="rgba(255,255,255,0.05)"),
        "shapes": [
            dict(type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                 line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot")),
            dict(type="line", x0=0, x1=1, y0=0.5, y1=0.5,
                 line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot")),
        ]}))
    f.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
        marker=dict(color=clrs, size=6, opacity=0.7), text=labels,
        hovertemplate="<b>%{text}</b><br>V=%{x:.2f} A=%{y:.2f}<extra></extra>"))
    if xs:
        f.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode="markers",
            marker=dict(color=ACCENT, size=14, symbol="star",
                        line=dict(color="white", width=1)),
            name="NOW"))
    return f

def fig_heatmap(frames: list) -> go.Figure:
    if len(frames) < 5: return _empty("🔥 EMOTION HEATMAP", 340)
    recent = frames[-50:]
    avg = {e: np.mean([fr["emotions"].get(e, 0) for fr in recent]) for e in FACE_48}
    top_emos = [k for k, _ in sorted(avg.items(), key=lambda x: -x[1])[:12]]
    matrix = np.array([[fr["emotions"].get(e, 0) for e in top_emos] for fr in recent]).T
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "🔥 EMOTION INTENSITY HEATMAP (last 50 frames)", "height": 340,
        "xaxis": dict(showgrid=False), "yaxis": dict(showgrid=False)}))
    f.add_trace(go.Heatmap(z=matrix, x=list(range(len(recent))), y=top_emos,
        colorscale=[[0,"#0d1620"],[0.3,"#1a4a6e"],[0.6,"#00b4d8"],[1,"#00e5ff"]],
        showscale=False))
    return f

def fig_facs(facs_dict: dict) -> go.Figure:
    if not facs_dict: return _empty("🦷 FACS ACTION UNITS")
    items  = sorted(facs_dict.items(), key=lambda x: -x[1])[:14]
    names  = [f"{k} — {FACS_NAMES.get(k,'?')}" for k, _ in items][::-1]
    scores = [v * 100 for _, v in items][::-1]
    f = go.Figure(layout=go.Layout(**{**_L, "title": "🦷 FACS ACTION UNITS", "height": 380,
        "xaxis": dict(title="Intensity %", gridcolor="rgba(255,255,255,0.05)"),
        "yaxis": dict(showgrid=False)}))
    f.add_trace(go.Bar(x=scores, y=names, orientation="h",
        marker=dict(color=WARN_C, opacity=0.8),
        text=[f"{s:.1f}%" for s in scores], textposition="outside"))
    return f

def fig_sentiment_dist(dist: list) -> go.Figure:
    sc = sentiment_score(dist)
    colors = ["#FF2020","#FF5A5A","#FF8080","#FFB347","#E0E0E0",
              "#90EE90","#5DCA5D","#2DFF7A","#00CC55"]
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": f"📊 SENTIMENT DISTRIBUTION  (avg: {sc:.1f}/9)", "height": 260,
        "xaxis": dict(title="Scale (1=most neg → 9=most pos)", showgrid=False),
        "yaxis": dict(title="%", gridcolor="rgba(255,255,255,0.05)")}))
    f.add_trace(go.Bar(
        x=[str(i+1) for i in range(9)], y=[v*100 for v in dist],
        marker=dict(color=colors, opacity=0.88),
        text=[f"{v*100:.1f}%" for v in dist], textposition="outside"))
    return f

def fig_toxicity(tox: dict) -> go.Figure:
    if not tox: return _empty("☠️ TOXICITY", 220)
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "☠️ TOXICITY DETECTION", "height": 240,
        "xaxis": dict(showgrid=False),
        "yaxis": dict(title="%", gridcolor="rgba(255,255,255,0.05)")}))
    f.add_trace(go.Bar(
        x=list(tox.keys()), y=[v*100 for v in tox.values()],
        marker=dict(color=NEG_C, opacity=0.8),
        text=[f"{v*100:.2f}%" for v in tox.values()], textposition="outside"))
    return f

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ CONFIG")

    api_key = st.text_input("🔑 Hume API Key", value=ss.api_key, type="password",
        placeholder="platform.hume.ai/settings/keys")
    ss.api_key = api_key

    mock_mode = st.checkbox("🧪 Mock Mode (no key needed)", value=(not bool(api_key)))

    if not HUME_OK and not mock_mode:
        st.error("`pip install hume`")

    st.markdown("---")
    run_live = st.toggle("▶ LIVE CAPTURE", value=ss.running)
    if run_live != ss.running:
        ss.running = run_live

    st.markdown("**Face Extras**")
    ss.enable_facs = st.checkbox("FACS action units",      value=ss.enable_facs)
    ss.enable_desc = st.checkbox("Facial descriptions",    value=ss.enable_desc)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        if st.button("🤖 Train ML", type="primary"):
            train_ml()
            st.toast(f"✅ {ss.ml_acc}% acc" if ss.ml_trained else f"Need {MIN_TRAIN}+ frames")
    with cb:
        if st.button("🗑️ Clear"):
            ss.face_frames.clear(); ss.lang_results.clear()
            ss.last_face = ss.last_lang = None
            ss.prev_img_id = None
            ss.ml_trained = False; ss.ml_acc = 0.0
            ss.ml_pipe = _build_pipe()
            st.rerun()

    st.markdown("---")
    st.metric("Face Frames",   len(ss.face_frames))
    st.metric("Lang Analyses", len(ss.lang_results))
    st.metric("ML Acc",        f"{ss.ml_acc}%" if ss.ml_trained else "⏳")

    if ss.last_face:
        st.markdown("**Last Frame**")
        for name, score in top_n(ss.last_face["emotions"], 5):
            c = emo_color(name)
            st.markdown(
                f'<div class="emo-row">'
                f'<div class="dot" style="background:{c}"></div>'
                f'<span style="color:{c};font-size:.76rem">{name}</span>'
                f'<span style="color:#555;font-size:.76rem;margin-left:auto">{score*100:.1f}%</span>'
                f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# ⬡ MULTIMODAL SENTIMENT — Hume AI")
st.markdown(
    f'<span class="tag" style="background:{ACCENT}18;color:{ACCENT};'
    f'border:1px solid {ACCENT}40">Face · 48 emotions + FACS + Descriptions</span>&nbsp;'
    f'<span class="tag" style="background:{POS_C}18;color:{POS_C};'
    f'border:1px solid {POS_C}40">Language · 53 emotions + Sentiment + Toxicity</span>',
    unsafe_allow_html=True)
st.markdown("")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Face dims",    "48 emotions")
h2.metric("Language dims","53 + sentiment + toxicity")
h3.metric("Mode",  "🧪 Mock" if mock_mode else "🔴 Hume API")
h4.metric("SDK",   "✅ hume installed" if HUME_OK else "❌ pip install hume")

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_face, tab_lang, tab_analytics = st.tabs(
    ["📷 Face Analysis", "💬 Language Analysis", "📊 Analytics"])

# ──────────────────────────────────────────────────────────────
#  TAB 1 — FACE
# ──────────────────────────────────────────────────────────────
with tab_face:
    st.markdown("### 📷 Real-Time Facial Expression via Hume")

    if ss.running:
        img_file = st.camera_input("Capture frame", key="cam_face")

        if img_file is not None:
            img_id = id(img_file)
            if img_id != ss.prev_img_id:
                ss.prev_img_id = img_id
                pil_img = Image.open(io.BytesIO(img_file.getvalue()))

                if mock_mode:
                    with st.spinner("🧪 Mock face analysis…"):
                        time.sleep(0.25)
                        result = mock_face()
                    st.caption("Mock mode — enter API key for real Hume analysis")
                else:
                    with st.spinner("🔍 Hume: analyzing 48 facial expressions…"):
                        result = call_hume_face(api_key, pil_img,
                                                facs=ss.enable_facs, desc=ss.enable_desc)

                if result:
                    result["ts"] = time.time()
                    ss.last_face = result
                    ss.face_frames.append(result)
                    if len(ss.face_frames) > MAX_HIST:
                        ss.face_frames.pop(0)
                    if len(ss.face_frames) >= MIN_TRAIN and len(ss.face_frames) % 25 == 0:
                        train_ml()

        if ss.last_face:
            emos = ss.last_face["emotions"]
            top_emo, top_sc = top_n(emos, 1)[0]
            c = emo_color(top_emo)
            p, n, u = valence(emos)
            v_label = "😊 POSITIVE" if p > n else ("😔 NEGATIVE" if n > p else "😐 NEUTRAL")
            st.markdown(
                f'<div style="background:{c}15;border:1px solid {c}40;border-radius:8px;'
                f'padding:10px 18px;margin-top:8px;display:flex;justify-content:space-between">'
                f'<span style="color:{c}">● {top_emo.upper()}  {top_sc*100:.1f}%</span>'
                f'<span style="color:#555">{v_label} · {len(ss.face_frames)} frames</span>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Click the camera button to capture a frame")
    else:
        n = len(ss.face_frames)
        st.info(
            "▶ Toggle **LIVE CAPTURE** in the sidebar.\n\n"
            "Hume face model: **48 emotions**, optional FACS action units & facial descriptions."
            if n == 0 else f"⏸ Paused — {n} frames. Toggle to resume.")

    # ── Face results ─────────────────────────────────────────
    if ss.last_face and PLOTLY_OK:
        st.markdown("---")
        emos = ss.last_face["emotions"]

        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(fig_bars(emos, "🎭 TOP 15 EMOTIONS — Current Frame"),
                            use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(fig_radar(emos),
                            use_container_width=True, config={"displayModeBar": False})

        p, n, u = valence(emos); t = p + n + u or 1
        v1, v2, v3 = st.columns(3)
        v1.metric("😊 Positive", f"{p/t*100:.1f}%")
        v2.metric("😔 Negative", f"{n/t*100:.1f}%")
        v3.metric("😐 Neutral",  f"{u/t*100:.1f}%")

        # FACS chart
        if ss.enable_facs and ss.last_face.get("facs"):
            st.markdown("---")
            st.plotly_chart(fig_facs(ss.last_face["facs"]),
                            use_container_width=True, config={"displayModeBar": False})

        # Descriptions
        if ss.enable_desc and ss.last_face.get("descriptions"):
            st.markdown("---")
            st.markdown("#### 🗒️ Facial Descriptions")
            descs = sorted(ss.last_face["descriptions"].items(), key=lambda x: -x[1])[:8]
            dc = st.columns(min(len(descs), 4))
            for i, (d, s) in enumerate(descs):
                dc[i % 4].metric(d, f"{s*100:.0f}%")

        # All 48 emotions grid
        st.markdown("---")
        st.markdown("#### 🎭 All 48 Emotions — Current Frame")
        all_sorted = sorted(emos.items(), key=lambda x: -x[1])
        gcols = st.columns(3)
        for i, (name, score) in enumerate(all_sorted):
            c = emo_color(name)
            gcols[i % 3].markdown(
                f'<div class="emo-row">'
                f'<div class="dot" style="background:{c}"></div>'
                f'<span style="color:#8fa0b0;font-size:.75rem;flex:1">{name}</span>'
                f'<span style="color:{c};font-size:.75rem;font-weight:bold">{score*100:.0f}%</span>'
                f'</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  TAB 2 — LANGUAGE
# ──────────────────────────────────────────────────────────────
with tab_lang:
    st.markdown("### 💬 Emotional Language Analysis")
    st.markdown(
        "Hume's language model returns **53 emotion scores**, "
        "a **9-point sentiment distribution**, and **6-category toxicity** detection.")

    text_input = st.text_area("Enter text to analyze", height=100,
        placeholder="e.g.  I can't believe how amazing this turned out — I'm genuinely thrilled!")

    if st.button("🔍 Analyze Text", type="primary"):
        if not text_input.strip():
            st.warning("Enter some text first.")
        else:
            if mock_mode:
                with st.spinner("🧪 Mock language analysis…"):
                    time.sleep(0.25)
                    lang_result = mock_language(text_input)
            elif not api_key:
                st.warning("Enter your Hume API key in the sidebar.")
                lang_result = None
            else:
                with st.spinner("🔍 Hume: analyzing emotional language…"):
                    lang_result = call_hume_language(api_key, text_input)

            if lang_result:
                lang_result["text"] = text_input
                lang_result["ts"]   = time.time()
                ss.last_lang = lang_result
                ss.lang_results.append(lang_result)
                if len(ss.lang_results) > MAX_HIST:
                    ss.lang_results.pop(0)

    if ss.last_lang and PLOTLY_OK:
        lr = ss.last_lang
        st.markdown("---")
        st.markdown(f"**Text:** *{lr['text'][:120]}{'…' if len(lr['text'])>120 else ''}*")

        sc = sentiment_score(lr["sentiment"])
        sc_color = POS_C if sc > 5.5 else (NEG_C if sc < 4.5 else NEU_C)
        st.markdown(
            f'<div style="background:{sc_color}15;border:1px solid {sc_color}40;'
            f'border-radius:8px;padding:10px 18px;margin:8px 0;display:inline-block">'
            f'<span style="color:{sc_color}">Sentiment: {sc:.1f}/9 '
            f'{"● Positive" if sc>5.5 else ("● Negative" if sc<4.5 else "● Neutral")}'
            f'</span></div>', unsafe_allow_html=True)

        lc1, lc2 = st.columns([3, 2])
        with lc1:
            st.plotly_chart(fig_bars(lr["emotions"], "🎭 TOP 15 LANGUAGE EMOTIONS"),
                            use_container_width=True, config={"displayModeBar": False})
        with lc2:
            st.plotly_chart(fig_sentiment_dist(lr["sentiment"]),
                            use_container_width=True, config={"displayModeBar": False})

        st.plotly_chart(fig_toxicity(lr["toxicity"]),
                        use_container_width=True, config={"displayModeBar": False})

        # Language-exclusive callout
        extras = {k: v for k, v in lr["emotions"].items() if k in LANG_EXTRAS}
        if extras:
            st.markdown("---")
            st.markdown("#### ✨ Language-Exclusive Emotions (not available in face model)")
            ec = st.columns(len(extras))
            for i, (name, score) in enumerate(sorted(extras.items(), key=lambda x: -x[1])):
                ec[i].metric(name.upper(), f"{score*100:.1f}%")

# ──────────────────────────────────────────────────────────────
#  TAB 3 — ANALYTICS
# ──────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Session Analytics")

    if not PLOTLY_OK:
        st.error("`pip install plotly`")
    elif len(ss.face_frames) < 2:
        st.info("Capture at least **2 face frames** in the 📷 tab to see analytics here.")
    else:
        # Valence timeline + arousal scatter
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(fig_valence_timeline(ss.face_frames),
                            use_container_width=True, config={"displayModeBar": False})
        with a2:
            st.plotly_chart(fig_arousal_scatter(ss.face_frames),
                            use_container_width=True, config={"displayModeBar": False})

        if len(ss.face_frames) >= 5:
            st.plotly_chart(fig_heatmap(ss.face_frames),
                            use_container_width=True, config={"displayModeBar": False})

        # Session stats
        st.markdown("---")
        st.markdown("#### 📐 Session Summary")
        avg_emos = {e: np.mean([fr["emotions"].get(e, 0) for fr in ss.face_frames])
                    for e in FACE_48}
        dom_emo, dom_sc = max(avg_emos.items(), key=lambda x: x[1])
        avg_ps = [valence(fr["emotions"])[0] for fr in ss.face_frames]
        avg_ns = [valence(fr["emotions"])[1] for fr in ss.face_frames]

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Frames",       len(ss.face_frames))
        s2.metric("Dominant",     dom_emo.upper())
        s3.metric("Dom Score",    f"{dom_sc*100:.1f}%")
        s4.metric("Avg Positive", f"{np.mean(avg_ps)*100:.1f}%")
        s5.metric("Avg Negative", f"{np.mean(avg_ns)*100:.1f}%")

        st.markdown("---")
        st.plotly_chart(fig_bars(avg_emos, "🏆 SESSION AVERAGE EMOTIONS"),
                        use_container_width=True, config={"displayModeBar": False})

        # ML status
        st.markdown("---")
        st.markdown("#### 🤖 ML Valence Classifier (RandomForest on 48-dim Hume vectors)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Status",   "✅ Trained" if ss.ml_trained else "⏳ Not trained")
        m2.metric("Accuracy", f"{ss.ml_acc}%" if ss.ml_trained else "—")
        m3.metric("Samples",  f"{len(ss.face_frames)} / {MIN_TRAIN} min")
        if not ss.ml_trained and len(ss.face_frames) >= MIN_TRAIN:
            st.info("👈 Click **🤖 Train ML** in the sidebar!")

        # Language sentiment trend
        if len(ss.lang_results) >= 2:
            st.markdown("---")
            st.markdown("#### 💬 Language Sentiment Trend")
            scores = [sentiment_score(lr["sentiment"]) for lr in ss.lang_results]
            labels = [lr["text"][:30] + "…" for lr in ss.lang_results]
            f = go.Figure(layout=go.Layout(**{**_L,
                "title": "💬 SENTIMENT SCORE OVER ANALYSES (1=neg, 9=pos)", "height": 260,
                "yaxis": dict(range=[1,9], gridcolor="rgba(255,255,255,0.05)"),
                "xaxis": dict(showgrid=False),
                "shapes": [dict(type="line", x0=0, x1=len(scores)-1, y0=5, y1=5,
                                line=dict(color=NEU_C, width=1, dash="dot"))]}))
            f.add_trace(go.Scatter(y=scores, mode="lines+markers", text=labels,
                line=dict(color=ACCENT, width=2),
                marker=dict(color=[POS_C if s>5 else NEG_C for s in scores], size=8),
                hovertemplate="<b>%{text}</b><br>Score: %{y:.1f}<extra></extra>"))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
