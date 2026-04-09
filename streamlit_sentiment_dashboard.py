"""
MULTIMODAL SENTIMENT DASHBOARD — v3
- st.camera_input() → browser webcam (works on Streamlit Cloud!)
- DeepFace analyzes each captured frame
- Mock mode if camera denied / unavailable
- Charts always render (use_container_width=True, not the broken width= param)
- Proper rerun control — no infinite loop when paused
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import io
from collections import Counter
from PIL import Image

# ── Page config MUST be first ──────────────────────────────
st.set_page_config(
    page_title="⬡ SENTIMENT LIVE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Graceful imports ────────────────────────────────────────
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
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ── Constants ───────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_IDX = {e: i for i, e in enumerate(EMOTIONS)}
EMOTION_EMOJI = {
    "angry": "😡", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😮"
}
EMOTION_COLORS = {
    "angry": "#FF3B3B", "disgust": "#20D4D4", "fear": "#C832C8",
    "happy": "#2DFF7A", "neutral": "#9A9A9A", "sad": "#5B8CFF", "surprise": "#FFB800",
}
BG, CARD_BG, ACCENT = "#050a0e", "#0d1620", "#00e5ff"
MIN_TRAIN, MAX_HISTORY = 20, 500
MOCK_WEIGHTS = [0.08, 0.03, 0.06, 0.25, 0.38, 0.12, 0.08]

# ── CSS ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .main .block-container {{ padding-top:1rem; padding-bottom:1rem; }}
  h1,h2,h3 {{ color:{ACCENT} !important; font-family:monospace; }}
  [data-testid="metric-container"] {{
    background:{CARD_BG}; border:1px solid rgba(0,229,255,0.25);
    border-radius:6px; padding:8px 12px;
  }}
  .emo-badge {{
    display:inline-block; padding:8px 24px; border-radius:20px;
    font-size:1.5rem; font-weight:bold; font-family:monospace; letter-spacing:2px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
_defaults = {
    "emotions":      [],
    "timestamps":    [],
    "probs_history": [],
    "ml_trained":    {k: False for k in ["LogReg", "RandForest", "SVM", "KNN"]},
    "ml_accuracy":   {k: 0.0   for k in ["LogReg", "RandForest", "SVM", "KNN"]},
    "ml_models":     {},
    "running":       False,
    "last_emotion":  "neutral",
    "last_probs":    [1/7]*7,
    "cam_mode":      "browser",
    "prev_img_id":   None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ss = st.session_state

# ── Migrate deque → list (old session state compatibility) ──
from collections import deque as _deque
for _k in ["emotions", "timestamps", "probs_history"]:
    if isinstance(ss.get(_k), _deque):
        ss[_k] = list(ss[_k])

# ── ML models ───────────────────────────────────────────────
def _build_models():
    if not SK_OK:
        return {}
    return {
        "LogReg":     Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]),
        "RandForest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60, random_state=42))]),
        "SVM":        Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
        "KNN":        Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }

if not ss.ml_models:
    ss.ml_models = _build_models()

# ── Mock helpers ─────────────────────────────────────────────
_mock_last = "neutral"
def _mock_emotion():
    global _mock_last
    if random.random() < 0.65:
        return _mock_last
    _mock_last = random.choices(EMOTIONS, weights=MOCK_WEIGHTS)[0]
    return _mock_last

def _mock_probs(emo):
    b = [0.02] * 7
    idx = EMOTION_IDX[emo]
    b[idx] = random.uniform(0.55, 0.85)
    rest = 1.0 - b[idx]
    for i in range(7):
        if i != idx:
            b[i] = rest / 6 * random.uniform(0.3, 1.7)
    t = sum(b)
    return [v/t for v in b]

# ── DeepFace analysis on PIL image ──────────────────────────
def _analyze_pil(pil_img):
    try:
        pil_img.thumbnail((320, 240), Image.LANCZOS)
        arr = np.array(pil_img.convert("RGB"))
        arr_bgr = arr[:, :, ::-1]
        if CV2_OK:
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        res = DeepFace.analyze(
            arr_bgr, actions=["emotion"],
            enforce_detection=False, silent=True,
            detector_backend="opencv"
        )
        raw = res[0]["emotion"]
        probs = [raw.get(e, 0) for e in EMOTIONS]
        total = sum(probs) or 1
        probs = [v/total for v in probs]
        emo = max(raw, key=raw.get).lower()
        return emo, probs
    except Exception:
        emo = "neutral"
        return emo, _mock_probs(emo)

# ── Record emotion ───────────────────────────────────────────
def _record(emo, probs):
    ss.last_emotion = emo
    ss.last_probs   = probs
    ss.emotions.append(emo)
    ss.timestamps.append(time.time())
    ss.probs_history.append(probs)
    if len(ss.emotions) > MAX_HISTORY:
        ss.emotions.pop(0)
        ss.timestamps.pop(0)
        ss.probs_history.pop(0)

# ── ML train ────────────────────────────────────────────────
def train_models():
    if not SK_OK or len(ss.emotions) < MIN_TRAIN:
        return
    X = np.array(ss.probs_history[-len(ss.emotions):])
    y = np.array([EMOTION_IDX[e] for e in ss.emotions])
    if len(np.unique(y)) < 2:
        return
    for nm, pipe in ss.ml_models.items():
        try:
            pipe.fit(X, y)
            ss.ml_trained[nm] = True
            ss.ml_accuracy[nm] = round(accuracy_score(y, pipe.predict(X)) * 100, 1)
        except Exception:
            pass

# ── Charts ───────────────────────────────────────────────────
_L = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font=dict(color="#b0b8c0", family="monospace", size=10),
    margin=dict(l=45, r=15, t=40, b=30),
    showlegend=False,
)

def _empty(title):
    f = go.Figure(layout=go.Layout(**{**_L, "title": title, "height": 280}))
    f.add_annotation(text="Collecting data...", xref="paper", yref="paper",
                     x=0.5, y=0.5, showarrow=False, font=dict(color="#444", size=13))
    return f

def chart_timeline():
    if not ss.emotions:
        return _empty("📊 EMOTION TIMELINE")
    recent = ss.emotions[-120:]
    y_vals = [EMOTION_IDX.get(e, 4) for e in recent]
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "📊 EMOTION TIMELINE", "height": 280,
        "xaxis": dict(showgrid=False, zeroline=False),
        "yaxis": dict(tickvals=list(range(7)),
                      ticktext=[e[:3].upper() for e in EMOTIONS],
                      gridcolor="rgba(255,255,255,0.05)"),
    }))
    f.add_trace(go.Scatter(y=y_vals, mode="lines", fill="tozeroy",
        line=dict(color=ACCENT, width=1.5), fillcolor="rgba(0,229,255,0.08)",
        text=recent, hovertemplate="<b>%{text}</b><extra></extra>"))
    f.add_trace(go.Scatter(y=y_vals, mode="markers",
        marker=dict(color=[EMOTION_COLORS[e] for e in recent], size=5, opacity=0.9),
        hoverinfo="skip"))
    return f

def chart_frequency():
    if not ss.emotions:
        return _empty("📈 FREQUENCY")
    c = Counter(ss.emotions)
    vals = [c.get(e, 0) for e in EMOTIONS]
    total = max(sum(vals), 1)
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "📈 FREQUENCY DISTRIBUTION", "height": 280,
        "xaxis": dict(showgrid=False),
        "yaxis": dict(gridcolor="rgba(255,255,255,0.05)"),
    }))
    f.add_trace(go.Bar(
        x=[e.upper() for e in EMOTIONS], y=vals,
        marker=dict(color=[EMOTION_COLORS[e] for e in EMOTIONS], opacity=0.85),
        text=[f"{round(v/total*100,1)}%" for v in vals], textposition="outside",
    ))
    return f

def chart_distribution():
    if not ss.emotions:
        return _empty("🍰 DISTRIBUTION")
    c = Counter(ss.emotions)
    pairs = [(e, c[e]) for e in EMOTIONS if c.get(e, 0) > 0]
    if not pairs:
        return _empty("🍰 DISTRIBUTION")
    labels, values = zip(*pairs)
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "🍰 DONUT DISTRIBUTION", "height": 280,
        "showlegend": True,
        "legend": dict(orientation="v", x=1.0, y=0.5, font=dict(size=9)),
    }))
    f.add_trace(go.Pie(
        labels=[l.upper() for l in labels], values=values, hole=0.55,
        marker=dict(colors=[EMOTION_COLORS[e] for e in labels], line=dict(width=0)),
        textinfo="percent",
    ))
    return f

def chart_model_accuracy():
    trained = {nm: acc for nm, acc in ss.ml_accuracy.items() if ss.ml_trained.get(nm)}
    if not trained:
        need = max(0, MIN_TRAIN - len(ss.emotions))
        msg = f"Need {need} more samples" if need > 0 else "Click 'Train Models Now'"
        f = go.Figure(layout=go.Layout(**{**_L, "title": "🤖 MODEL ACCURACY", "height": 280}))
        f.add_annotation(text=msg, xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, font=dict(color="#444", size=13))
        return f
    f = go.Figure(layout=go.Layout(**{**_L,
        "title": "🤖 ML MODEL ACCURACY", "height": 280,
        "yaxis": dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)"),
        "xaxis": dict(showgrid=False),
    }))
    colors = ["#00e5ff", "#2DFF7A", "#FFB800", "#C832C8"]
    f.add_trace(go.Bar(
        x=list(trained.keys()), y=list(trained.values()),
        marker=dict(color=colors[:len(trained)], opacity=0.85),
        text=[f"{v}%" for v in trained.values()], textposition="outside",
    ))
    return f

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ CONTROLS")

    run_live = st.toggle("▶ LIVE CAPTURE", value=ss.running)
    if run_live != ss.running:
        ss.running = run_live

    cam_choice = st.radio(
        "Camera Source",
        ["📷 Browser Camera (Real)", "🖥️ Mock (Simulated)"],
        index=0 if ss.cam_mode == "browser" else 1,
    )
    ss.cam_mode = "browser" if "Browser" in cam_choice else "mock"

    st.markdown("---")

    if st.button("🔄 Train Models Now", type="primary"):
        train_models()
        ready = sum(ss.ml_trained.values())
        st.toast(f"✅ {ready}/4 models trained!" if ready else "⚠️ Need 20+ samples first")

    if st.button("🗑️ Clear Data"):
        ss.emotions.clear(); ss.timestamps.clear(); ss.probs_history.clear()
        ss.ml_trained  = {k: False for k in ss.ml_trained}
        ss.ml_accuracy = {k: 0.0   for k in ss.ml_accuracy}
        ss.ml_models   = _build_models()
        ss.prev_img_id = None
        st.rerun()

    st.markdown("---")
    st.metric("Samples", len(ss.emotions))
    st.metric("Models Ready", f"{sum(ss.ml_trained.values())} / 4")

    emo = ss.last_emotion
    col = EMOTION_COLORS.get(emo, "#fff")
    st.markdown("**Last Detected**")
    st.markdown(
        f'<div class="emo-badge" style="background:{col}22;color:{col};'
        f'border:1px solid {col}55">'
        f'{EMOTION_EMOJI.get(emo,"")} {emo.upper()}</div>',
        unsafe_allow_html=True
    )
    if ss.last_probs and any(p > 0 for p in ss.last_probs):
        st.markdown("**Confidence**")
        for e, p in sorted(zip(EMOTIONS, ss.last_probs), key=lambda x: -x[1])[:4]:
            st.progress(int(p * 100), text=f"{e[:4].upper()} {p*100:.0f}%")

# ════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════
st.markdown("# ⬡ MULTIMODAL SENTIMENT INTELLIGENCE")
st.markdown("*Real-time facial emotion → ML classification → live analytics*")

if ss.running:

    if ss.cam_mode == "browser":
        # ── BROWSER CAMERA ─────────────────────────────────
        # st.camera_input works on Streamlit Cloud — it accesses YOUR browser camera
        st.markdown("#### 📷 Point your face at the camera and click the capture button")
        img_file = st.camera_input(label="Capture frame", key="cam_widget")

        if img_file is not None:
            img_id = id(img_file)
            if img_id != ss.prev_img_id:
                ss.prev_img_id = img_id
                pil_img = Image.open(io.BytesIO(img_file.getvalue()))

                if DEEPFACE_OK:
                    with st.spinner("🔍 Analyzing emotion..."):
                        emo, probs = _analyze_pil(pil_img)
                else:
                    emo = _mock_emotion()
                    probs = _mock_probs(emo)
                    st.warning("DeepFace not available — using mock detection")

                _record(emo, probs)

                n = len(ss.emotions)
                if n >= MIN_TRAIN and n % 30 == 0:
                    train_models()

            n = len(ss.emotions)
            emo = ss.last_emotion
            c = EMOTION_COLORS.get(emo, "#fff")
            st.markdown(
                f'<div style="background:{c}22;border:1px solid {c}55;border-radius:6px;'
                f'padding:10px 18px;font-family:monospace;color:{c};font-size:1.1rem;margin-top:8px">'
                f'🟢 {EMOTION_EMOJI.get(emo,"")} <b>{emo.upper()}</b> &nbsp;·&nbsp; '
                f'{n} samples &nbsp;·&nbsp; {sum(ss.ml_trained.values())}/4 models</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("👆 Click the camera button above — allow camera access if the browser asks.\n\n"
                    "Each click captures one frame for analysis.")

    else:
        # ── MOCK MODE ──────────────────────────────────────
        for _ in range(3):
            emo = _mock_emotion()
            _record(emo, _mock_probs(emo))

        n = len(ss.emotions)
        if n >= MIN_TRAIN and n % 30 == 0:
            train_models()

        emo = ss.last_emotion
        c = EMOTION_COLORS.get(emo, "#fff")
        st.markdown(
            f"""<div style="background:{CARD_BG};border:1px solid {c}44;border-radius:10px;
                     padding:30px;text-align:center;max-width:460px;margin:0 auto">
              <div style="font-size:.7rem;color:#555;letter-spacing:3px;margin-bottom:8px">
                SIMULATED STREAM · SAMPLE #{n}
              </div>
              <div style="font-size:3.5rem;margin:10px 0">{EMOTION_EMOJI.get(emo,'')}</div>
              <div class="emo-badge" style="font-size:1.8rem;background:{c}22;
                   color:{c};border:2px solid {c}66;padding:8px 28px">{emo.upper()}</div>
              <div style="font-size:.65rem;color:#444;margin-top:12px">
                Switch to "Browser Camera" in sidebar for real detection
              </div>
            </div>""",
            unsafe_allow_html=True
        )
        time.sleep(0.4)
        st.rerun()   # mock auto-streams; browser cam is event-driven (no auto-rerun needed)

else:
    n = len(ss.emotions)
    if n == 0:
        st.info("▶ Toggle **LIVE CAPTURE** in the sidebar, then choose your camera source.\n\n"
                "**Browser Camera** works directly on Streamlit Cloud — no local install needed!")
    else:
        st.info(f"⏸ Paused — {n} samples collected. Toggle to resume.")

# ════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 LIVE ANALYTICS")

if not PLOTLY_OK:
    st.error("Plotly not installed — run: pip install plotly")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_timeline(),      use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(chart_frequency(),     use_container_width=True, config={"displayModeBar": False})
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_distribution(),  use_container_width=True, config={"displayModeBar": False})
    with c4:
        st.plotly_chart(chart_model_accuracy(), use_container_width=True, config={"displayModeBar": False})

# ════════════════════════════════════════════════════════════
# MODEL STATUS + STATS
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🤖 MODEL STATUS")

cols = st.columns(4)
for nm, col in zip(["LogReg", "RandForest", "SVM", "KNN"], cols):
    col.metric(nm, f"✅ {ss.ml_accuracy[nm]}%" if ss.ml_trained.get(nm) else "⏳ Waiting")

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
    s3.metric("Dominance %",      f"{round(dominant[1]/total*100,1)}%")
    s4.metric("Entropy (bits)",   f"{entropy:.2f}")
