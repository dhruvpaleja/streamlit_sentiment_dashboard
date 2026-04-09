"""
MULTIMODAL SENTIMENT DASHBOARD — FIXED & FAST
- Auto-detects webcam; falls back to realistic mock mode
- Proper rerun control (no infinite loop)
- DeepFace on real frames (every 5th frame to stay fast)
- ML models with actual multi-feature vectors
- 4 charts that always render
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
from collections import Counter

# ── Page config MUST be first ──────────────────────────────
st.set_page_config(
    page_title="⬡ SENTIMENT LIVE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Imports with graceful fallbacks ────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    SK_OK = True
except ImportError:
    SK_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ── Constants ───────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_IDX = {e: i for i, e in enumerate(EMOTIONS)}
EMOTION_COLORS = {
    "angry":    "#FF3B3B",
    "disgust":  "#20D4D4",
    "fear":     "#C832C8",
    "happy":    "#2DFF7A",
    "neutral":  "#9A9A9A",
    "sad":      "#5B8CFF",
    "surprise": "#FFB800",
}

BG       = "#050a0e"
CARD_BG  = "#0d1620"
ACCENT   = "#00e5ff"
MIN_TRAIN = 20
MAX_HISTORY = 500

# ── Realistic emotion distribution (for mock mode) ─────────
MOCK_WEIGHTS = [0.08, 0.03, 0.06, 0.25, 0.38, 0.12, 0.08]  # neutral/happy dominate

# ── CSS ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .main .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
  h1, h2, h3 {{ color: {ACCENT} !important; font-family: monospace; }}
  [data-testid="metric-container"] {{
      background: {CARD_BG};
      border: 1px solid rgba(0,229,255,0.25);
      border-radius: 6px;
      padding: 8px 12px;
  }}
  .emotion-badge {{
      display: inline-block;
      padding: 6px 18px;
      border-radius: 20px;
      font-size: 1.4rem;
      font-weight: bold;
      font-family: monospace;
      letter-spacing: 2px;
  }}
  .stButton button {{
      background: {CARD_BG};
      color: {ACCENT};
      border: 1px solid {ACCENT};
      border-radius: 4px;
  }}
</style>
""", unsafe_allow_html=True)

# ── One-time camera detection (run before session init) ─────
def _detect_mock_mode() -> bool:
    """Check once if webcam + DeepFace are available. Suppress OpenCV stderr spam."""
    if not CV2_OK or not DEEPFACE_OK:
        return True
    import os, sys
    # Suppress the torrent of OpenCV camera-not-found error messages
    devnull = open(os.devnull, "w")
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
    except Exception:
        ok = False
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        devnull.close()
    return not ok

# ── Session state init ──────────────────────────────────────
def init_state():
    defaults = {
        "emotions":      [],
        "timestamps":    [],
        "probs_history": [],   # list of 7-dim probability vectors
        "ml_trained":    {k: False for k in ["LogReg", "RandForest", "SVM", "KNN"]},
        "ml_accuracy":   {k: 0.0   for k in ["LogReg", "RandForest", "SVM", "KNN"]},
        "ml_models":     {},
        "running":       False,
        "frame_count":   0,
        "mock_mode":     _detect_mock_mode(),
        "last_emotion":  "neutral",
        "last_probs":    [0]*7,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
ss = st.session_state

# ── Build ML models dict ─────────────────────────────────────
def get_models():
    if not SK_OK:
        return {}
    return {
        "LogReg":     Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500, C=1.0))]),
        "RandForest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60, random_state=42))]),
        "SVM":        Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
        "KNN":        Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }

if not ss.ml_models:
    ss.ml_models = get_models()

# ── Mock emotion generator (realistic temporal correlation) ──
_last_mock = "neutral"
def mock_emotion():
    global _last_mock
    # 70% chance to stay same or drift to nearby emotion (makes timeline look natural)
    stay_prob = 0.6
    if random.random() < stay_prob:
        return _last_mock
    _last_mock = random.choices(EMOTIONS, weights=MOCK_WEIGHTS)[0]
    return _last_mock

def mock_probs(emotion: str) -> list:
    base = [0.02] * 7
    idx = EMOTION_IDX[emotion]
    base[idx] = random.uniform(0.55, 0.85)
    # sprinkle noise
    rest = 1.0 - base[idx]
    for i in range(7):
        if i != idx:
            base[i] = rest / 6 * random.uniform(0.3, 1.7)
    total = sum(base)
    return [v / total for v in base]

# ── Real frame analysis ──────────────────────────────────────
def analyze_frame(frame):
    try:
        small = cv2.resize(frame, (128, 96), interpolation=cv2.INTER_NEAREST)
        res = DeepFace.analyze(
            small, actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="opencv"
        )
        probs_raw = res[0]["emotion"]  # dict {emotion: float}
        probs_ordered = [probs_raw.get(e, 0) for e in EMOTIONS]
        total = sum(probs_ordered) or 1
        probs_norm = [v / total for v in probs_ordered]
        emotion = max(probs_raw, key=probs_raw.get).lower()
        return emotion, probs_norm
    except Exception:
        return "neutral", mock_probs("neutral")

# ── ML training (multi-feature: full prob vector) ────────────
def train_models():
    if not SK_OK or len(ss.emotions) < MIN_TRAIN:
        return
    if not ss.probs_history or len(ss.probs_history) != len(ss.emotions):
        return

    X = np.array(ss.probs_history)            # shape (N, 7)
    y = np.array([EMOTION_IDX[e] for e in ss.emotions])

    if len(np.unique(y)) < 2:
        return

    for name, pipe in ss.ml_models.items():
        try:
            pipe.fit(X, y)
            ss.ml_trained[name] = True
            preds = pipe.predict(X)
            ss.ml_accuracy[name] = round(accuracy_score(y, preds) * 100, 1)
        except Exception:
            pass

# ── Chart helpers ─────────────────────────────────────────────
_layout_base = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color="#b0b8c0", family="monospace", size=10),
    margin=dict(l=45, r=15, t=40, b=30),
    showlegend=False,
)

def _fig(**kwargs):
    layout = {**_layout_base, **kwargs}
    return go.Figure(layout=go.Layout(**layout))

def chart_timeline():
    n = len(ss.emotions)
    if n == 0:
        f = _fig(title="📊 EMOTION TIMELINE", height=280)
        f.add_annotation(text="Collecting data...", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, font=dict(color="#555", size=14))
        return f

    recent = ss.emotions[-120:]
    y_vals = [EMOTION_IDX.get(e, 4) for e in recent]
    colors = [EMOTION_COLORS[e] for e in recent]

    f = _fig(title="📊 EMOTION TIMELINE", height=280,
             xaxis=dict(showgrid=False, zeroline=False),
             yaxis=dict(tickvals=list(range(7)),
                        ticktext=[e[:3].upper() for e in EMOTIONS],
                        gridcolor="rgba(255,255,255,0.05)"))

    # Filled area
    f.add_trace(go.Scatter(
        y=y_vals, mode="lines",
        line=dict(color=ACCENT, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.08)",
        hovertemplate="<b>%{text}</b><extra></extra>",
        text=recent,
    ))

    # Scatter dots colored by emotion
    f.add_trace(go.Scatter(
        y=y_vals, mode="markers",
        marker=dict(color=colors, size=5, opacity=0.9),
        hoverinfo="skip",
    ))
    return f

def chart_frequency():
    if not ss.emotions:
        f = _fig(title="📈 FREQUENCY", height=280)
        f.add_annotation(text="Collecting data...", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, font=dict(color="#555", size=14))
        return f

    counts = Counter(ss.emotions)
    vals = [counts.get(e, 0) for e in EMOTIONS]
    pct  = [round(v / max(sum(vals), 1) * 100, 1) for v in vals]

    f = _fig(title="📈 FREQUENCY DISTRIBUTION", height=280,
             xaxis=dict(showgrid=False),
             yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
    f.add_trace(go.Bar(
        x=[e.upper() for e in EMOTIONS],
        y=vals,
        marker=dict(color=[EMOTION_COLORS[e] for e in EMOTIONS], opacity=0.85),
        text=[f"{p}%" for p in pct],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    return f

def chart_distribution():
    if not ss.emotions:
        f = _fig(title="🍰 DISTRIBUTION", height=280)
        f.add_annotation(text="Collecting data...", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, font=dict(color="#555", size=14))
        return f

    counts = Counter(ss.emotions)
    vals  = [counts.get(e, 0) for e in EMOTIONS]
    nonzero = [(e, v) for e, v in zip(EMOTIONS, vals) if v > 0]
    if not nonzero:
        return _fig(title="🍰 DISTRIBUTION", height=280)

    labels, values = zip(*nonzero)
    f = _fig(title="🍰 DONUT DISTRIBUTION", height=280,
             showlegend=True,
             legend=dict(orientation="v", x=1, y=0.5, font=dict(size=9)))
    f.add_trace(go.Pie(
        labels=[l.upper() for l in labels],
        values=values,
        hole=0.55,
        marker=dict(colors=[EMOTION_COLORS[e] for e in labels], line=dict(width=0)),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value} samples (%{percent})<extra></extra>",
    ))
    return f

def chart_model_accuracy():
    trained = {nm: acc for nm, acc in ss.ml_accuracy.items() if ss.ml_trained.get(nm)}

    if not trained:
        f = _fig(title="🤖 MODEL ACCURACY", height=280)
        need = max(0, MIN_TRAIN - len(ss.emotions))
        msg = f"Need {need} more samples" if need > 0 else "Click 'Train Models'"
        f.add_annotation(text=msg, xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, font=dict(color="#555", size=14))
        return f

    colors = ["#00e5ff", "#2DFF7A", "#FFB800", "#C832C8"]
    f = _fig(title="🤖 ML MODEL ACCURACY", height=280,
             yaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)"),
             xaxis=dict(showgrid=False))
    f.add_trace(go.Bar(
        x=list(trained.keys()),
        y=list(trained.values()),
        marker=dict(color=colors[:len(trained)], opacity=0.85),
        text=[f"{v}%" for v in trained.values()],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
    ))
    return f

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ CONTROLS")

    run_live = st.toggle("▶ LIVE CAPTURE", value=ss.running, key="run_toggle")
    if run_live != ss.running:
        ss.running = run_live

    if st.button("🔄 Train Models Now", type="primary"):
        train_models()
        st.toast("✅ Models trained!" if any(ss.ml_trained.values()) else "⚠️ Need more data")

    if st.button("🗑️ Clear Data"):
        ss.emotions.clear()
        ss.timestamps.clear()
        ss.probs_history.clear()
        ss.ml_trained = {k: False for k in ss.ml_trained}
        ss.ml_accuracy = {k: 0.0   for k in ss.ml_accuracy}
        ss.ml_models = get_models()
        st.rerun()

    st.markdown("---")
    st.metric("Samples",       len(ss.emotions))
    st.metric("Models Ready",  f"{sum(ss.ml_trained.values())} / 4")
    mode_label = "🖥️ Mock (Cloud)" if ss.mock_mode else "📷 Webcam (Local)"
    st.caption(f"Mode: {mode_label}")
    if ss.mock_mode:
        st.caption("_No webcam detected — using simulated emotion stream_")

    st.markdown("---")
    st.markdown("**Last Emotion**")
    emo = ss.last_emotion
    col = EMOTION_COLORS.get(emo, "#fff")
    st.markdown(
        f'<div class="emotion-badge" style="background:{col}22;color:{col};border:1px solid {col}55">'
        f'{emo.upper()}</div>',
        unsafe_allow_html=True
    )

    if ss.probs_history and ss.last_probs:
        st.markdown("**Confidence**")
        probs_df = pd.DataFrame({
            "emotion": [e[:4].upper() for e in EMOTIONS],
            "conf":    [round(p * 100, 1) for p in ss.last_probs]
        }).sort_values("conf", ascending=False).head(4)
        for _, row in probs_df.iterrows():
            st.progress(int(row["conf"]), text=f"{row['emotion']} {row['conf']:.0f}%")

# ── Header ─────────────────────────────────────────────────
st.markdown("# ⬡ MULTIMODAL SENTIMENT INTELLIGENCE")
st.markdown("*Real-time facial emotion → ML classification → live analytics*")

# ── Live capture section ────────────────────────────────────
frame_placeholder = st.empty()
status_placeholder = st.empty()

if ss.running:
    if ss.mock_mode:
        # ── MOCK MODE ──────────────────────────────────────
        # Generate 3 emotion readings per rerun (fast accumulation)
        for _ in range(3):
            emo = mock_emotion()
            probs = mock_probs(emo)
            ss.emotions.append(emo)
            ss.timestamps.append(time.time())
            ss.probs_history.append(probs)
            ss.last_emotion = emo
            ss.last_probs = probs
            if len(ss.emotions) > MAX_HISTORY:
                ss.emotions.pop(0)
                ss.timestamps.pop(0)
                ss.probs_history.pop(0)

        # Show a live "emotion card" instead of webcam
        emo = ss.last_emotion
        col = EMOTION_COLORS.get(emo, "#fff")
        n = len(ss.emotions)
        frame_placeholder.markdown(
            f"""
            <div style="background:{CARD_BG};border:1px solid {col}44;border-radius:10px;
                        padding:30px;text-align:center;max-width:480px;margin:0 auto">
              <div style="font-size:0.75rem;color:#666;letter-spacing:3px;margin-bottom:8px">
                LIVE DETECTION · SAMPLE #{n}
              </div>
              <div style="font-size:3.5rem;margin:10px 0">
                {'😡' if emo=='angry' else '🤢' if emo=='disgust' else '😨' if emo=='fear'
                 else '😊' if emo=='happy' else '😐' if emo=='neutral'
                 else '😢' if emo=='sad' else '😮'}
              </div>
              <div class="emotion-badge" style="font-size:1.8rem;background:{col}22;
                           color:{col};border:2px solid {col}66;padding:8px 28px">
                {emo.upper()}
              </div>
              <div style="font-size:0.65rem;color:#444;margin-top:12px">
                Mock stream · deploy locally for real webcam
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        # ── REAL WEBCAM MODE ───────────────────────────────
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            ss.mock_mode = True
            status_placeholder.warning("Webcam unavailable — switched to mock mode")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            for _ in range(6):   # grab 6 frames per rerun burst
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                ss.frame_count += 1

                # DeepFace every 5th frame only
                if ss.frame_count % 5 == 0:
                    emo, probs = analyze_frame(frame)
                    ss.last_emotion = emo
                    ss.last_probs   = probs
                    ss.emotions.append(emo)
                    ss.timestamps.append(time.time())
                    ss.probs_history.append(probs)
                    if len(ss.emotions) > MAX_HISTORY:
                        ss.emotions.pop(0)
                        ss.timestamps.pop(0)
                        ss.probs_history.pop(0)

                # Draw overlay
                emo = ss.last_emotion
                col_bgr = tuple(int(EMOTION_COLORS[emo].lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
                cv2.putText(frame, emo.upper(), (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, col_bgr, 3)
                cv2.putText(frame, f"#{len(ss.emotions)}", (20, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    width="stretch"
                )

            cap.release()

    # Auto-train every 30 new samples
    n = len(ss.emotions)
    if n >= MIN_TRAIN and n % 30 == 0:
        train_models()

    status_placeholder.success(
        f"🟢 LIVE  ·  {len(ss.emotions)} samples  ·  "
        f"models: {sum(ss.ml_trained.values())}/4 trained"
    )

    time.sleep(0.4)
    st.rerun()   # only rerun when running=True — no infinite loop when paused

else:
    if len(ss.emotions) == 0:
        frame_placeholder.info("▶ Toggle **LIVE CAPTURE** in the sidebar to start.")
    else:
        frame_placeholder.info(f"⏸ Paused. {len(ss.emotions)} samples collected.")

# ── Analytics charts ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 LIVE ANALYTICS")

if not PLOTLY_OK:
    st.error("Plotly not installed — `pip install plotly`")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_timeline(),    width="stretch", config={"displayModeBar": False})
    with c2:
        st.plotly_chart(chart_frequency(),   width="stretch", config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_distribution(), width="stretch", config={"displayModeBar": False})
    with c4:
        st.plotly_chart(chart_model_accuracy(), width="stretch", config={"displayModeBar": False})

# ── Model status row ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 MODEL STATUS")

cols = st.columns(4)
for i, (nm, col) in enumerate(zip(["LogReg", "RandForest", "SVM", "KNN"], cols)):
    trained = ss.ml_trained.get(nm, False)
    acc     = ss.ml_accuracy.get(nm, 0)
    col.metric(nm, f"✅ {acc}%" if trained else "⏳ Waiting")

# ── Stats row ────────────────────────────────────────────────
if len(ss.emotions) >= 5:
    st.markdown("---")
    st.markdown("### 📐 SESSION STATS")
    counts = Counter(ss.emotions)
    dominant = counts.most_common(1)[0]
    total = len(ss.emotions)
    entropy = -sum((v/total) * np.log2(v/total + 1e-9) for v in counts.values())

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Samples",    total)
    s2.metric("Dominant Emotion", dominant[0].upper())
    s3.metric("Dominance %",      f"{round(dominant[1]/total*100, 1)}%")
    s4.metric("Entropy (bits)",   f"{entropy:.2f}")
