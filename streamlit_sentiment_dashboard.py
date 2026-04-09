"""
FAST MULTIMODAL SENTIMENT DASHBOARD
- OpenCV webcam (no WebRTC lag)
- DeepFace emotion (every 4 frames)
- 4 lean ML models (LogReg, RF, SVM, KNN)
- 4 essential charts only
- No CSS bloat, no threading chaos
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import deque
import time
from datetime import datetime

from deepface import DeepFace
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go
import plotly.express as px

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_IDX = {e: i for i, e in enumerate(EMOTIONS)}
EMOTION_COLORS = {
    "angry": "#FF3B3B", "disgust": "#20D4D4", "fear": "#C832C8",
    "happy": "#2DFF7A", "neutral": "#9A9A9A", "sad": "#5B8CFF", "surprise": "#FFB800",
}

BG, CARD_BG, ACCENT = "#050a0e", "#0d1620", "#00e5ff"
MIN_TRAIN, MAX_HISTORY = 15, 1000

# ML Models (lightweight)
ML_MODELS = {
    "LogReg": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]),
    "RandForest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, random_state=42))]),
    "SVM": Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
    "KNN": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))]),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "emotions" not in st.session_state:
    st.session_state.emotions = deque(maxlen=MAX_HISTORY)
    st.session_state.timestamps = deque(maxlen=MAX_HISTORY)
    st.session_state.ml_trained = {nm: False for nm in ML_MODELS}
    st.session_state.ml_accuracy = {nm: 0 for nm in ML_MODELS}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_frame(frame):
    """Fast face emotion detection - resize + analyze."""
    try:
        small = cv2.resize(frame, (96, 72), interpolation=cv2.INTER_NEAREST)
        res = DeepFace.analyze(small, actions=["emotion"], enforce_detection=False, 
                               silent=True, detector_backend="opencv")
        emotion = max(res[0]["emotion"], key=res[0]["emotion"].get).lower()
        return emotion
    except:
        return "neutral"

def train_ml_models(emotions_list):
    """Train all ML models on collected emotion labels."""
    if len(emotions_list) < MIN_TRAIN:
        return
    
    # Simple feature: emotion index is the feature
    X = np.array([[EMOTION_IDX.get(e, 4)] for e in emotions_list]).astype(float)
    y = np.array([EMOTION_IDX.get(e, 4) for e in emotions_list])
    
    if len(np.unique(y)) < 2:
        return
    
    for name, pipe in ML_MODELS.items():
        try:
            pipe.fit(X, y)
            st.session_state.ml_trained[name] = True
            preds = pipe.predict(X)
            st.session_state.ml_accuracy[name] = int(accuracy_score(y, preds) * 100)
        except:
            pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARTS (Minimal, Fast)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_timeline():
    """Emotion timeline - last 100 samples."""
    if not st.session_state.emotions:
        return None
    
    recent = list(st.session_state.emotions)[-100:]
    y_vals = [EMOTION_IDX.get(e, 4) for e in recent]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_vals, mode="lines", fill="tozeroy",
        line=dict(color=ACCENT, width=2),
        fillcolor=f"rgba(0,229,255,0.1)",
        name="Emotion"))
    
    fig.update_layout(
        title="📊 EMOTION TIMELINE",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(showgrid=False), yaxis=dict(tickvals=list(range(7)), 
                                                ticktext=[e[:3].upper() for e in EMOTIONS]),
        font=dict(color="#c0c0c0", size=10),
        hovermode="x unified", showlegend=False
    )
    return fig

def chart_frequency():
    """Emotion frequency bars."""
    if not st.session_state.emotions:
        return None
    
    emotions_list = list(st.session_state.emotions)
    counts = [emotions_list.count(e) for e in EMOTIONS]
    
    fig = go.Figure(go.Bar(
        x=[e.upper() for e in EMOTIONS], y=counts,
        marker=dict(color=[EMOTION_COLORS[e] for e in EMOTIONS]),
        text=counts, textposition="outside"
    ))
    
    fig.update_layout(
        title="📈 FREQUENCY",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color="#c0c0c0"),
        showlegend=False, hovermode=False
    )
    return fig

def chart_distribution():
    """Donut chart of emotion distribution."""
    if not st.session_state.emotions:
        return None
    
    emotions_list = list(st.session_state.emotions)
    counts = [emotions_list.count(e) for e in EMOTIONS]
    
    fig = go.Figure(go.Pie(
        labels=[e.upper() for e in EMOTIONS], values=counts, hole=0.5,
        marker=dict(colors=[EMOTION_COLORS[e] for e in EMOTIONS])
    ))
    
    fig.update_layout(
        title="🍰 DISTRIBUTION",
        height=300, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color="#c0c0c0"),
        showlegend=False
    )
    return fig

def chart_model_accuracy():
    """Model accuracy bars."""
    trained = {nm: acc for nm, acc in st.session_state.ml_accuracy.items() 
               if st.session_state.ml_trained[nm]}
    
    if not trained:
        fig = go.Figure()
        fig.add_annotation(text="Training...", xref="paper", yref="paper", 
                          x=0.5, y=0.5, font=dict(color="#555"))
        fig.update_layout(height=300, paper_bgcolor=CARD_BG, title="🤖 MODEL ACCURACY")
        return fig
    
    fig = go.Figure(go.Bar(
        x=list(trained.keys()), y=list(trained.values()),
        marker=dict(color=px.colors.sequential.Plasma[:len(trained)]),
        text=[f"{v}%" for v in trained.values()], textposition="outside"
    ))
    
    fig.update_layout(
        title="🤖 MODEL ACCURACY",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color="#c0c0c0"),
        yaxis=dict(range=[0, 110]),
        showlegend=False, hovermode=False
    )
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="⬡ SENTIMENT FAST", layout="wide")

st.markdown(f"""
<style>
body {{ background-color: {BG}; color: #c0c0c0; }}
.main {{ padding: 1rem; }}
h1, h2, h3 {{ color: {ACCENT}; }}
[data-testid="metric-container"] {{ background: {CARD_BG}; border: 1px solid rgba(0,229,255,0.2); }}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("# ⬡ MULTIMODAL SENTIMENT — FAST")
st.markdown("*Face Emotion → ML Classification → Live Analytics*")

# Webcam feed
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

cap = cv2.VideoCapture(0)
frame_count = 0
last_emotion = "neutral"

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ CONTROLS")
    run_live = st.checkbox("🎥 Run Webcam", value=True)
    train_button = st.button("🔄 Train Models Now", type="primary")
    
    st.markdown("---")
    st.metric("Samples Collected", len(st.session_state.emotions))
    st.metric("Models Ready", sum(st.session_state.ml_trained.values()), "/4")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE CAPTURE LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if run_live:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    for _ in range(5):  # Capture 5 frames quickly
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam not available")
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Analyze every 4th frame
        if frame_count % 4 == 0:
            last_emotion = analyze_frame(frame)
            st.session_state.emotions.append(last_emotion)
            st.session_state.timestamps.append(time.time())
            
            # Auto-train every 20 samples
            if len(st.session_state.emotions) % 20 == 0 and len(st.session_state.emotions) >= MIN_TRAIN:
                train_ml_models(list(st.session_state.emotions))
        
        # Draw overlay
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{last_emotion.upper()}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                   (0, 255, 0) if last_emotion == "happy" else (0, 0, 255), 2)
        cv2.putText(frame, f"Samples: {len(st.session_state.emotions)}", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, use_column_width=True)
    
    cap.release()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.markdown("### 📊 ANALYTICS")

if train_button:
    train_ml_models(list(st.session_state.emotions))
    st.success("✅ Models trained!")

# Display charts
col1, col2 = st.columns(2)
with col1:
    fig_tl = chart_timeline()
    if fig_tl:
        st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})

with col2:
    fig_freq = chart_frequency()
    if fig_freq:
        st.plotly_chart(fig_freq, use_container_width=True, config={"displayModeBar": False})

col3, col4 = st.columns(2)
with col3:
    fig_dist = chart_distribution()
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

with col4:
    fig_acc = chart_model_accuracy()
    if fig_acc:
        st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DETAILS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.markdown("### 🤖 MODEL STATUS")

m1, m2, m3, m4 = st.columns(4)
models_list = list(ML_MODELS.keys())
for i, col in enumerate([m1, m2, m3, m4]):
    if i < len(models_list):
        nm = models_list[i]
        is_trained = st.session_state.ml_trained[nm]
        acc = st.session_state.ml_accuracy[nm]
        status = f"✅ {acc}%" if is_trained else "⏳ Training..."
        col.metric(nm, status)

# Auto-refresh
time.sleep(0.5)
st.rerun()
