"""
MULTIMODAL SENTIMENT DASHBOARD — v4 (Hume AI Edition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upgrade from DeepFace (7 emotions) → Hume AI (48 emotions)

What's new vs v3:
  ✦ Hume Expression Measurement streaming API (48 FACS emotions)
  ✦ Valence analysis: Positive / Negative / Neutral balance timeline
  ✦ Radar chart for emotion profile snapshot
  ✦ Full emotion breakdown with progress bars
  ✦ Arousal × Valence quadrant scatter plot
  ✦ Mock mode — works without API key for testing
  ✦ sklearn ML layer still trains on top of Hume embeddings

Install:
  pip install streamlit hume plotly scikit-learn pillow

Run:
  streamlit run streamlit_sentiment_hume_v4.py
"""

import streamlit as st
import asyncio
import io
import numpy as np
import time
import random
from collections import Counter
from PIL import Image

# ── Page config (MUST be first) ─────────────────────────────
st.set_page_config(
    page_title="⬡ HUME SENTIMENT LIVE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Graceful imports ─────────────────────────────────────────
try:
    from hume import AsyncHumeClient
    from hume.expression_measurement.stream.socket_client import StreamConnectOptions
    HUME_OK = True
except ImportError:
    HUME_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    SK_OK = True
except ImportError:
    SK_OK = False

# ── Hume's 48 Emotion Labels ─────────────────────────────────
# Hume returns these in title-case; we lowercase for consistency
ALL_48_EMOTIONS = [
    "admiration", "adoration", "aesthetic appreciation", "amusement", "anger",
    "anxiety", "awe", "awkwardness", "boredom", "calmness", "concentration",
    "confusion", "contempt", "contentment", "craving", "curiosity", "desire",
    "determination", "disappointment", "disgust", "distress", "doubt", "ecstasy",
    "embarrassment", "empathic pain", "enthusiasm", "entrancement", "envy",
    "excitement", "fear", "guilt", "horror", "interest", "joy", "love",
    "nostalgia", "pain", "pride", "realization", "relief", "romance", "sadness",
    "satisfaction", "shame", "surprise (negative)", "surprise (positive)",
    "sympathy", "tiredness", "triumph",
]

# Valence groupings
POSITIVE = {
    "admiration", "adoration", "aesthetic appreciation", "amusement", "awe",
    "calmness", "contentment", "curiosity", "desire", "determination",
    "ecstasy", "enthusiasm", "entrancement", "excitement", "interest",
    "joy", "love", "pride", "relief", "romance", "satisfaction",
    "surprise (positive)", "triumph",
}
NEGATIVE = {
    "anger", "anxiety", "awkwardness", "boredom", "contempt", "craving",
    "disappointment", "disgust", "distress", "doubt", "embarrassment",
    "empathic pain", "envy", "fear", "guilt", "horror", "nostalgia",
    "pain", "sadness", "shame", "surprise (negative)", "sympathy", "tiredness",
}
# Anything not in above → neutral (concentration, confusion, realization, adoration overlap)

# Arousal mapping (high energy vs calm) — approximate
HIGH_AROUSAL = {
    "anger", "excitement", "ecstasy", "fear", "horror", "triumph", "enthusiasm",
    "amusement", "surprise (positive)", "surprise (negative)", "distress", "awe",
}
LOW_AROUSAL = {
    "calmness", "tiredness", "boredom", "contentment", "satisfaction",
    "nostalgia", "sadness", "disappointment",
}

# ── Color Palette ─────────────────────────────────────────────
BG       = "#050a0e"
CARD_BG  = "#0d1620"
ACCENT   = "#00e5ff"
POS_CLR  = "#2DFF7A"
NEG_CLR  = "#FF3B3B"
NEU_CLR  = "#9A9A9A"
WARN_CLR = "#FFB800"

def emotion_color(name: str) -> str:
    if name in POSITIVE: return POS_CLR
    if name in NEGATIVE: return NEG_CLR
    return NEU_CLR

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  html, body, [class*="css"] {{ font-family: 'Share Tech Mono', monospace; }}
  .main .block-container {{ padding-top:0.8rem; padding-bottom:1rem; }}
  h1,h2,h3 {{ color:{ACCENT} !important; font-family:'Share Tech Mono',monospace; }}
  [data-testid="metric-container"] {{
    background:{CARD_BG}; border:1px solid rgba(0,229,255,0.2);
    border-radius:6px; padding:8px 12px;
  }}
  .emo-pill {{
    display:inline-block; padding:6px 18px; border-radius:20px;
    font-size:1.1rem; font-weight:bold; letter-spacing:2px;
    font-family:'Share Tech Mono',monospace;
  }}
  .hume-badge {{
    background:rgba(0,229,255,0.08); border:1px solid rgba(0,229,255,0.3);
    border-radius:8px; padding:6px 14px; font-size:0.75rem; color:{ACCENT};
    display:inline-block; margin-bottom:8px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
_defaults = {
    "frames":            [],   # list of dicts: {emotions: {name: score}, ts: float}
    "running":           False,
    "last_frame":        None,
    "prev_img_id":       None,
    "api_key":           "",
    "use_mock":          False,
    "ml_models":         {},
    "ml_trained":        False,
    "ml_accuracy":       0.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
ss = st.session_state

MAX_HISTORY = 300
MIN_TRAIN   = 20

# ── Hume API call ──────────────────────────────────────────────
async def _hume_analyze(api_key: str, img_bytes: bytes) -> dict:
    """
    Sends a JPEG frame to Hume Expression Measurement streaming API.
    Returns a dict of {emotion_name_lower: score_float} for all 48 emotions.

    Hume streaming response shape (face model):
      result.face.predictions[0].emotions → list of EmotionScore(name, score)
    """
    client = AsyncHumeClient(api_key=api_key)

    # StreamConnectOptions with face model enabled
    # If your SDK version differs, try: Config(face=FaceConfig()) from hume.expression_measurement.stream
    options = StreamConnectOptions()   # defaults to all models; face included

    async with client.expression_measurement.stream.connect(options=options) as socket:
        # send_file accepts raw bytes; model config can be passed per-payload
        # Pass models param so only face is computed (faster + cheaper)
        result = await socket.send_file(
            img_bytes,
            # Uncomment if your SDK version supports per-payload config:
            # models={"face": {}}
        )

    # Parse response
    emotions = {}
    try:
        preds = result.face.predictions
        if preds:
            for emo_score in preds[0].emotions:
                name = emo_score.name.lower()
                emotions[name] = float(emo_score.score)
    except Exception:
        # Fallback: try dict-style access
        try:
            raw = result["face"]["predictions"][0]["emotions"]
            for item in raw:
                emotions[item["name"].lower()] = float(item["score"])
        except Exception:
            pass

    return emotions


def run_hume(api_key: str, pil_img: Image.Image) -> dict | None:
    """Wrapper to run async Hume call synchronously from Streamlit."""
    try:
        pil_img.thumbnail((640, 480), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_hume_analyze(api_key, img_bytes))
        loop.close()
        return result if result else None
    except Exception as e:
        st.error(f"Hume API error: {e}")
        return None

# ── Mock Hume result ───────────────────────────────────────────
_mock_state = {"last_top": ["joy", "calmness"]}

def mock_hume_result() -> dict:
    """Generates realistic mock Hume scores for testing without API key."""
    emotions = {e: random.uniform(0.01, 0.15) for e in ALL_48_EMOTIONS}

    # Temporal smoothing — slowly drift between states
    if random.random() < 0.25:
        _mock_state["last_top"] = random.choices(ALL_48_EMOTIONS, k=random.randint(1, 3))

    for top in _mock_state["last_top"]:
        emotions[top] = random.uniform(0.45, 0.82)

    total = sum(emotions.values())
    return {k: v / total for k, v in emotions.items()}

# ── Helpers ────────────────────────────────────────────────────
def valence_split(emo_dict: dict) -> tuple[float, float, float]:
    pos = sum(v for k, v in emo_dict.items() if k in POSITIVE)
    neg = sum(v for k, v in emo_dict.items() if k in NEGATIVE)
    neu = max(0, 1.0 - pos - neg)
    return pos, neg, neu

def arousal_score(emo_dict: dict) -> float:
    hi = sum(v for k, v in emo_dict.items() if k in HIGH_AROUSAL)
    lo = sum(v for k, v in emo_dict.items() if k in LOW_AROUSAL)
    return (hi - lo + 1) / 2  # 0=very calm, 1=very activated

def top_n(emo_dict: dict, n: int = 10) -> list[tuple[str, float]]:
    return sorted(emo_dict.items(), key=lambda x: -x[1])[:n]

# ── Chart helpers ──────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color="#8fa0b0", family="'Share Tech Mono',monospace", size=10),
    margin=dict(l=40, r=20, t=40, b=30),
    showlegend=False,
)

def _empty_fig(title: str, height: int = 280):
    f = go.Figure(layout=go.Layout(**{**_LAYOUT, "title": title, "height": height}))
    f.add_annotation(text="Waiting for data…", xref="paper", yref="paper",
                     x=0.5, y=0.5, showarrow=False,
                     font=dict(color="#334", size=13))
    return f


def fig_top_bars(emo_dict: dict) -> go.Figure:
    """Horizontal bar — top 15 emotions from current frame."""
    items = top_n(emo_dict, 15)
    names = [i[0] for i in items][::-1]
    scores = [i[1] * 100 for i in items][::-1]
    colors = [emotion_color(n) for n in names]

    f = go.Figure(layout=go.Layout(**{**_LAYOUT,
        "title": "🎭 CURRENT FRAME — TOP EMOTIONS",
        "height": 380,
        "xaxis": dict(title="Score %", gridcolor="rgba(255,255,255,0.05)"),
        "yaxis": dict(showgrid=False),
    }))
    f.add_trace(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{s:.1f}%" for s in scores], textposition="outside",
    ))
    return f


def fig_radar(emo_dict: dict) -> go.Figure:
    """Radar/spider chart across 12 representative emotions."""
    cats = [
        "joy", "excitement", "anger", "fear", "sadness", "disgust",
        "awe", "calmness", "curiosity", "amusement", "determination", "anxiety",
    ]
    vals = [emo_dict.get(c, 0) * 100 for c in cats]

    f = go.Figure(layout=go.Layout(**{
        **_LAYOUT,
        "title": "🕸️ EMOTION RADAR",
        "height": 380,
        "polar": dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 70],
                            gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(size=8)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ),
    }))
    closed_vals = vals + [vals[0]]
    closed_cats = cats + [cats[0]]
    f.add_trace(go.Scatterpolar(
        r=closed_vals, theta=closed_cats,
        fill="toself",
        fillcolor="rgba(0,229,255,0.12)",
        line=dict(color=ACCENT, width=2),
        mode="lines+markers",
        marker=dict(color=ACCENT, size=5),
    ))
    return f


def fig_valence_timeline() -> go.Figure:
    """Stacked area chart: positive / negative / neutral over time."""
    if len(ss.frames) < 2:
        return _empty_fig("📊 VALENCE OVER TIME")

    recent = ss.frames[-100:]
    pos_s, neg_s, neu_s = [], [], []
    for fr in recent:
        p, n, u = valence_split(fr["emotions"])
        pos_s.append(p * 100)
        neg_s.append(n * 100)
        neu_s.append(u * 100)

    x = list(range(len(pos_s)))
    f = go.Figure(layout=go.Layout(**{**_LAYOUT,
        "title": "📊 VALENCE OVER TIME",
        "height": 280,
        "showlegend": True,
        "legend": dict(orientation="h", y=1.15),
        "yaxis": dict(title="%", gridcolor="rgba(255,255,255,0.05)", range=[0, 110]),
        "xaxis": dict(title="Frame #", showgrid=False),
    }))
    f.add_trace(go.Scatter(x=x, y=pos_s, name="Positive",
                           fill="tozeroy", mode="lines",
                           line=dict(color=POS_CLR, width=1.5),
                           fillcolor="rgba(45,255,122,0.12)"))
    f.add_trace(go.Scatter(x=x, y=neg_s, name="Negative",
                           fill="tozeroy", mode="lines",
                           line=dict(color=NEG_CLR, width=1.5),
                           fillcolor="rgba(255,59,59,0.10)"))
    f.add_trace(go.Scatter(x=x, y=neu_s, name="Neutral",
                           fill="tozeroy", mode="lines",
                           line=dict(color=NEU_CLR, width=1),
                           fillcolor="rgba(154,154,154,0.07)"))
    return f


def fig_valence_arousal_scatter() -> go.Figure:
    """2-D valence × arousal scatter — each point is a frame."""
    if len(ss.frames) < 3:
        return _empty_fig("⚡ VALENCE × AROUSAL SPACE")

    recent = ss.frames[-120:]
    xs, ys, labels, colors_ = [], [], [], []
    for fr in recent:
        p, n, _ = valence_split(fr["emotions"])
        valence  = (p - n + 1) / 2          # 0 = very negative, 1 = very positive
        arousal  = arousal_score(fr["emotions"])
        xs.append(valence)
        ys.append(arousal)
        top_emo  = max(fr["emotions"], key=fr["emotions"].get)
        labels.append(top_emo)
        colors_.append(emotion_color(top_emo))

    f = go.Figure(layout=go.Layout(**{**_LAYOUT,
        "title": "⚡ VALENCE × AROUSAL (Russell Circumplex)",
        "height": 320,
        "xaxis": dict(title="← Negative  |  Positive →",
                      range=[0, 1], gridcolor="rgba(255,255,255,0.05)",
                      zeroline=False),
        "yaxis": dict(title="← Calm  |  Activated →",
                      range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
        "shapes": [
            # quadrant dividers
            dict(type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                 line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot")),
            dict(type="line", x0=0, x1=1, y0=0.5, y1=0.5,
                 line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot")),
        ],
    }))
    f.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(color=colors_, size=6, opacity=0.75),
        text=labels,
        hovertemplate="<b>%{text}</b><br>valence=%{x:.2f} arousal=%{y:.2f}<extra></extra>",
    ))
    # Highlight most recent
    if xs:
        f.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]],
            mode="markers",
            marker=dict(color=ACCENT, size=14, symbol="star",
                        line=dict(color="white", width=1)),
            text=[labels[-1]],
            hovertemplate="<b>NOW: %{text}</b><extra></extra>",
        ))
    return f


def fig_emotion_heatmap() -> go.Figure:
    """Heatmap of top-10 emotions across last 50 frames."""
    if len(ss.frames) < 5:
        return _empty_fig("🔥 EMOTION HEATMAP", 300)

    recent = ss.frames[-50:]
    # Pick top 10 emotions by average score
    avg_scores = {}
    for e in ALL_48_EMOTIONS:
        avg_scores[e] = np.mean([fr["emotions"].get(e, 0) for fr in recent])
    top_emos = [k for k, _ in sorted(avg_scores.items(), key=lambda x: -x[1])[:10]]

    matrix = np.array([[fr["emotions"].get(e, 0) for e in top_emos] for fr in recent]).T

    f = go.Figure(layout=go.Layout(**{**_LAYOUT,
        "title": "🔥 EMOTION INTENSITY HEATMAP (last 50 frames)",
        "height": 320,
        "xaxis": dict(title="Frame →", showgrid=False),
        "yaxis": dict(showgrid=False),
    }))
    f.add_trace(go.Heatmap(
        z=matrix,
        x=list(range(len(recent))),
        y=top_emos,
        colorscale=[
            [0.0,  "#0d1620"],
            [0.3,  "#1a4a6e"],
            [0.6,  "#00b4d8"],
            [1.0,  "#00e5ff"],
        ],
        showscale=False,
    ))
    return f

# ── ML layer (trains on Hume embeddings → predicts valence class) ──
def _build_ml():
    if not SK_OK: return {}
    return {
        "LogReg":     Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=400))]),
        "RandForest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60))]),
    }

if not ss.ml_models:
    ss.ml_models = _build_ml()

def train_ml():
    if not SK_OK or len(ss.frames) < MIN_TRAIN: return
    X, y = [], []
    for fr in ss.frames:
        vec = [fr["emotions"].get(e, 0) for e in ALL_48_EMOTIONS]
        p, n, _ = valence_split(fr["emotions"])
        label = 0 if p >= n else 1    # 0=positive, 1=negative
        X.append(vec)
        y.append(label)
    if len(set(y)) < 2: return
    X, y = np.array(X), np.array(y)
    accs = []
    for nm, pipe in ss.ml_models.items():
        try:
            pipe.fit(X, y)
            accs.append(accuracy_score(y, pipe.predict(X)) * 100)
        except Exception:
            pass
    ss.ml_trained  = True
    ss.ml_accuracy = round(np.mean(accs), 1) if accs else 0.0

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ CONTROLS")

    # API Key
    api_key = st.text_input(
        "🔑 Hume API Key",
        value=ss.api_key, type="password",
        placeholder="hume_…",
        help="Get a free key at platform.hume.ai → Expression Measurement",
    )
    ss.api_key = api_key

    if not HUME_OK:
        st.warning("Hume SDK missing:\n`pip install hume`", icon="⚠️")

    mock_mode = st.checkbox(
        "🧪 Mock Mode (no API key needed)",
        value=(not bool(api_key)),
    )

    st.markdown("---")

    run_live = st.toggle("▶ LIVE CAPTURE", value=ss.running)
    if run_live != ss.running:
        ss.running = run_live

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🤖 Train ML", type="primary"):
            train_ml()
            if ss.ml_trained:
                st.toast(f"✅ Avg accuracy: {ss.ml_accuracy}%")
            else:
                st.toast(f"Need {MIN_TRAIN}+ frames first")
    with col_b:
        if st.button("🗑️ Clear"):
            ss.frames.clear()
            ss.last_frame = None
            ss.prev_img_id = None
            ss.ml_trained  = False
            ss.ml_accuracy = 0.0
            ss.ml_models   = _build_ml()
            st.rerun()

    st.markdown("---")
    st.metric("Frames Analyzed", len(ss.frames))
    st.metric("ML Valence Acc", f"{ss.ml_accuracy}%" if ss.ml_trained else "⏳")

    # Last frame top emotions
    if ss.last_frame:
        st.markdown("**Top Emotions Now**")
        for name, score in top_n(ss.last_frame["emotions"], 5):
            clr = emotion_color(name)
            pct = int(score * 100)
            st.markdown(
                f'<div style="background:{clr}15;border-left:3px solid {clr};'
                f'padding:3px 8px;margin:2px 0;color:{clr};font-size:.8rem">'
                f'{name} — {pct}%</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# ⬡ MULTIMODAL SENTIMENT — Hume AI")
st.markdown(
    '<div class="hume-badge">Powered by Hume Expression Measurement · 48-emotion FACS model</div>',
    unsafe_allow_html=True,
)
col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Emotion Dimensions", "48" if (HUME_OK or mock_mode) else "⚠️ Install hume")
col_h2.metric("Mode", "🧪 Mock" if mock_mode else "🔴 Live Hume API")
col_h3.metric("SDK", "✅ hume" if HUME_OK else "❌ Missing")

# ══════════════════════════════════════════════════════════════
#  CAMERA + ANALYSIS
# ══════════════════════════════════════════════════════════════
st.markdown("---")

if ss.running:
    if not mock_mode and not HUME_OK:
        st.error("Install the Hume SDK first: `pip install hume`")
        st.stop()
    if not mock_mode and not api_key:
        st.warning("Enter your Hume API key in the sidebar, or enable Mock Mode.")

    st.markdown("#### 📷 Capture a frame to analyze")
    img_file = st.camera_input("Capture frame", key="cam_widget")

    if img_file is not None:
        img_id = id(img_file)

        if img_id != ss.prev_img_id:
            ss.prev_img_id = img_id
            pil_img = Image.open(io.BytesIO(img_file.getvalue()))

            if mock_mode:
                with st.spinner("🧪 Generating mock analysis…"):
                    time.sleep(0.3)
                    emotions = mock_hume_result()
                st.caption("Mock mode — enable Hume API for real analysis")
            else:
                with st.spinner("🔍 Hume AI: analyzing 48 facial emotions…"):
                    emotions = run_hume(api_key, pil_img)

            if emotions:
                frame_data = {"emotions": emotions, "ts": time.time()}
                ss.last_frame = frame_data
                ss.frames.append(frame_data)
                if len(ss.frames) > MAX_HISTORY:
                    ss.frames.pop(0)

                # Auto-train every 30 frames
                if len(ss.frames) % 30 == 0 and len(ss.frames) >= MIN_TRAIN:
                    train_ml()

        # ── Status bar ──
        if ss.last_frame:
            emo_dict  = ss.last_frame["emotions"]
            top_emo, top_score = top_n(emo_dict, 1)[0]
            clr = emotion_color(top_emo)
            p, n, u = valence_split(emo_dict)
            valence_label = "😊 POSITIVE" if p > n else ("😔 NEGATIVE" if n > p else "😐 NEUTRAL")

            st.markdown(
                f'<div style="background:{clr}15;border:1px solid {clr}40;'
                f'border-radius:8px;padding:12px 20px;font-family:monospace;'
                f'display:flex;justify-content:space-between;align-items:center;margin-top:8px">'
                f'<span style="color:{clr};font-size:1.1rem">● {top_emo.upper()} ({top_score*100:.1f}%)</span>'
                f'<span style="color:#666">{valence_label} · {len(ss.frames)} frames</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("👆 Click the camera button · allow browser camera access if prompted\n\n"
                "Each click = one frame analyzed through Hume's 48-emotion model.")
else:
    n = len(ss.frames)
    msg = ("▶ Toggle **LIVE CAPTURE** in the sidebar to begin.\n\n"
           "**Hume Expression Measurement** returns 48 fine-grained emotions — "
           "joy, awe, entrancement, determination, awkwardness, and 43 more."
           if n == 0 else
           f"⏸ Paused — {n} frames collected. Toggle to resume.")
    st.info(msg)

# ══════════════════════════════════════════════════════════════
#  CHARTS — CURRENT FRAME
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 LIVE ANALYTICS")

if not PLOTLY_OK:
    st.error("Install plotly: `pip install plotly`")
elif ss.last_frame:
    emo_dict = ss.last_frame["emotions"]

    # Row 1: bars + radar
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(fig_top_bars(emo_dict), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.plotly_chart(fig_radar(emo_dict), use_container_width=True,
                        config={"displayModeBar": False})

    # Valence meter
    p, n, u = valence_split(emo_dict)
    total = p + n + u or 1
    st.markdown("#### 🌡️ Valence Breakdown")
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("😊 Positive", f"{p/total*100:.1f}%")
    vc2.metric("😔 Negative", f"{n/total*100:.1f}%")
    vc3.metric("😐 Neutral",  f"{u/total*100:.1f}%")

    # Row 2: timeline + arousal
    if len(ss.frames) >= 2:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(fig_valence_timeline(), use_container_width=True,
                            config={"displayModeBar": False})
        with r2:
            st.plotly_chart(fig_valence_arousal_scatter(), use_container_width=True,
                            config={"displayModeBar": False})

    # Row 3: heatmap
    if len(ss.frames) >= 5:
        st.plotly_chart(fig_emotion_heatmap(), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Full emotion dump (all 48) ──────────────────────────
    st.markdown("---")
    st.markdown("### 🎭 ALL 48 EMOTIONS — Current Frame")

    sorted_all = sorted(emo_dict.items(), key=lambda x: -x[1])
    cols = st.columns(3)
    for i, (emo, score) in enumerate(sorted_all):
        clr = emotion_color(emo)
        pct = int(score * 100)
        with cols[i % 3]:
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:3px 0;gap:8px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{clr};flex-shrink:0"></div>'
                f'<span style="color:#8fa0b0;font-size:.78rem;flex:1">{emo}</span>'
                f'<span style="color:{clr};font-size:.78rem;font-weight:bold">{pct}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

else:
    st.info("🎯 Capture your first frame to see the Hume emotion breakdown.")

# ── ML Status ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 ML LAYER — Valence Classifier (trained on Hume embeddings)")
ml1, ml2, ml3 = st.columns(3)
ml1.metric("Status",   "✅ Trained" if ss.ml_trained else "⏳ Waiting")
ml2.metric("Accuracy", f"{ss.ml_accuracy}%" if ss.ml_trained else "—")
ml3.metric("Samples",  f"{len(ss.frames)} / {MIN_TRAIN} needed")

if not ss.ml_trained and len(ss.frames) >= MIN_TRAIN:
    st.info("You have enough data! Click **Train ML** in the sidebar.")

# ── Session stats ────────────────────────────────────────────
if len(ss.frames) >= 5:
    st.markdown("---")
    st.markdown("### 📐 SESSION STATS")
    avg_emotions = {}
    for e in ALL_48_EMOTIONS:
        avg_emotions[e] = np.mean([fr["emotions"].get(e, 0) for fr in ss.frames])
    dom_emo, dom_score = max(avg_emotions.items(), key=lambda x: x[1])

    avg_pos, avg_neg, avg_neu = [], [], []
    for fr in ss.frames:
        p, n, u = valence_split(fr["emotions"])
        avg_pos.append(p); avg_neg.append(n); avg_neu.append(u)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Frames",       len(ss.frames))
    s2.metric("Dominant Emotion",   dom_emo.upper())
    s3.metric("Avg Positive %",     f"{np.mean(avg_pos)*100:.1f}%")
    s4.metric("Avg Negative %",     f"{np.mean(avg_neg)*100:.1f}%")
