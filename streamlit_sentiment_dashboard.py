"""
╔══════════════════════════════════════════════════════════════╗
║   MULTIMODAL SENTIMENT — LIVE STREAMLIT DASHBOARD           ║
║   Face + Voice + Text | 6 ML Models | Live Plotly Charts    ║
║   Run: streamlit run streamlit_sentiment_dashboard.py        ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import streamlit as st
import cv2
import numpy as np
import threading
import time
import datetime
import warnings
import os
from collections import deque
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (cohen_kappa_score, confusion_matrix,
                                     accuracy_score, classification_report)
from sklearn.pipeline        import Pipeline
from scipy.stats             import entropy as scipy_entropy
import librosa
import speech_recognition as sr
from deepface import DeepFace
try:
    DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=["emotion"], enforce_detection=False, silent=True)
except:
    pass

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
EMOTIONS        = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_IDX     = {e: i for i, e in enumerate(EMOTIONS)}
N_EMOTIONS      = len(EMOTIONS)
FACE_WEIGHT     = 50.0
TONAL_WEIGHT    = 25.0
TEXT_WEIGHT     = 25.0
MIN_TRAIN       = 20
REFRESH_RATE    = 0.4   # seconds between UI refresh

EMOTION_COLORS = {
    "angry":   "#FF3B3B",
    "disgust": "#20D4D4",
    "fear":    "#C832C8",
    "happy":   "#2DFF7A",
    "neutral": "#9A9A9A",
    "sad":     "#5B8CFF",
    "surprise":"#FFB800",
}

BG       = "#050a0e"
CARD_BG  = "#0d1620"
ACCENT   = "#00e5ff"
ACCENT2  = "#7B61FF"
GRID_COL = "rgba(255,255,255,0.06)"

HF_LABEL_MAP = {
    "POSITIVE":"happy","NEGATIVE":"sad","NEUTRAL":"neutral",
    "POS":"happy","NEG":"sad","NEU":"neutral",
    "LABEL_0":"sad","LABEL_1":"neutral","LABEL_2":"happy",
}

# ─────────────────────────────────────────────────────────────────
# GLOBAL SHARED STATE  (written by threads, read by Streamlit)
# ─────────────────────────────────────────────────────────────────
MAX_HISTORY = 5000   # FIX #4: cap all session lists to prevent RAM leak

_lock = threading.Lock()
_state = {
    "frame":        None,
    "face_e":       "neutral",
    "tonal_e":      "neutral",
    "text_e":       "neutral",
    "final_e":      "neutral",
    "ml_preds":     {nm: "—" for nm in ["LogReg","RandForest","SVM","MLP","KNN","NaiveBayes"]},
    "transcript":   "",
    "session": {k: deque(maxlen=MAX_HISTORY) for k in [
        "timestamps","face_seq","tonal_seq","text_seq","final_seq",
        "face_str","tonal_str","text_str","final_str",
    ]},
    "feature_store":  deque(maxlen=MAX_HISTORY),   # FIX #4: bounded deque
    "ml_trained":     {nm: False for nm in ["LogReg","RandForest","SVM","MLP","KNN","NaiveBayes"]},
    "ml_cv_scores":   {nm: [] for nm in ["LogReg","RandForest","SVM","MLP","KNN","NaiveBayes"]},
    "audio_active":   False,
    "is_training":    False,   # FIX #3: concurrency guard — only one training thread at a time
    "ui_run_audio":   True,    # synced from sidebar toggle each render
    "ui_run_webcam":  True,    # synced from sidebar toggle each render
    "weights":        {"face": 50.0, "tonal": 25.0, "text": 25.0},  # FIX #1: synced from sliders
    "frame_count":    0,
    "fps":            0.0,
    "_fps_t":         time.time(),
    "_fps_n":         0,
}

# ─────────────────────────────────────────────────────────────────
# ML PIPELINES  (module-level so threads can access)
# ─────────────────────────────────────────────────────────────────
ML_MODELS = {
    "LogReg":     Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))]),
    "RandForest": Pipeline([("sc", StandardScaler()),
                             ("clf", RandomForestClassifier(n_estimators=100,random_state=42))]),
    "SVM":        Pipeline([("sc", StandardScaler()),
                             ("clf", SVC(kernel="rbf",C=1.0,probability=True))]),
    "MLP":        Pipeline([("sc", StandardScaler()),
                             ("clf", MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,random_state=42))]),
    "KNN":        Pipeline([("sc", StandardScaler()),
                             ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "NaiveBayes": Pipeline([("sc", StandardScaler()),
                             ("clf", GaussianNB())]),
}
_ml_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — loads once per session)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading Whisper...")
def load_whisper():
    import whisper
    return whisper.load_model("base")

@st.cache_resource(show_spinner="⚡ Loading RoBERTa sentiment...")
def load_text_pipeline():
    import torch
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1,
    )

# ─────────────────────────────────────────────────────────────────
# FEATURE + STAT HELPERS
# ─────────────────────────────────────────────────────────────────
def build_fv(face_e, tonal_e, text_e, fc=1.0, tc=1.0, xc=1.0):
    return np.array([
        EMOTION_IDX.get(face_e,  4),
        EMOTION_IDX.get(tonal_e, 4),
        EMOTION_IDX.get(text_e,  4),
        float(fc), float(tc), float(xc),
    ], dtype=float)

def weighted_fusion(face, tonal, text, weights):
    scores = {e: 0.0 for e in EMOTIONS}
    if face  in EMOTIONS: scores[face]  += weights["face"]
    if tonal in EMOTIONS: scores[tonal] += weights["tonal"]
    if text  in EMOTIONS: scores[text]  += weights["text"]
    return max(scores, key=scores.get)

def emotion_entropy(seq):
    if not seq: return 0.0
    counts = np.bincount(seq, minlength=N_EMOTIONS).astype(float)
    probs  = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))

def modality_corr(a, b):
    if len(a) < 5: return 0.0
    a, b = np.array(a[-100:], float), np.array(b[-100:], float)
    if a.std() == 0 or b.std() == 0: return 0.0
    val = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if (np.isnan(val) or np.isinf(val)) else val   # FIX #5: NaN → 0.0

def kappa(a, b):
    if len(a) < 5: return 0.0
    try:    return cohen_kappa_score(a[-100:], b[-100:])
    except: return 0.0

# ─────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────
_emotion_history = deque(maxlen=10)

def analyze_face(frame):
    try:
        # FIX #4: No lazy import — DeepFace already imported & built at module level
        res = DeepFace.analyze(frame, actions=["emotion"],
                               enforce_detection=False, silent=True)
        dom = max(res[0]["emotion"], key=res[0]["emotion"].get).lower()
        _emotion_history.append(dom)
        return max(set(_emotion_history), key=_emotion_history.count)
    except Exception as e:
        # FIX #4: Log real errors instead of silently returning neutral
        print(f"[DeepFace] {e}")
        return "neutral"

def extract_voice_tonality(audio_data, sample_rate=16000):
    try:
        if audio_data.max() > 0:
            audio_data = audio_data / np.abs(audio_data).max()
        energy = float(np.mean(librosa.feature.rms(y=audio_data)))
        pitch  = librosa.yin(audio_data, fmin=50, fmax=400)
        mp_    = float(np.mean(pitch[pitch > 0])) if np.any(pitch > 0) else 0
        mfcc   = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40), axis=1)
        if energy > 0.07 and mp_ > 250: return "angry"
        if energy > 0.05 and mp_ > 180: return "happy"
        if energy < 0.01 and mp_ < 130: return "sad"
        if mfcc[1] > 10:                return "happy"
        return "neutral"
    except:
        return "neutral"

def analyze_text(text, text_pipeline):
    if not text or len(text.strip()) < 2: return "neutral"
    try:
        r = text_pipeline(text[:512])[0]
        return HF_LABEL_MAP.get(r["label"].upper(), "neutral")
    except:
        return "neutral"

def predict_all_models(face_e, tonal_e, text_e):
    fv = build_fv(face_e, tonal_e, text_e).reshape(1, -1)
    preds = {}
    with _ml_lock:
        for name, pipe in ML_MODELS.items():
            if _state["ml_trained"][name]:
                try:
                    idx = pipe.predict(fv)[0]
                    preds[name] = EMOTIONS[idx]
                except:
                    preds[name] = "neutral"
            else:
                preds[name] = "—"
    return preds

def train_all_models():
    # 1. Consistent snapshot under primary lock
    with _lock:
        current_X = list(_state["feature_store"])
        current_y = list(_state["session"]["final_seq"])

    try:
        # 2. Slice + validate — early returns are now safe because finally always runs
        MAX_TRAIN = 2000
        if len(current_X) > MAX_TRAIN:
            current_X = current_X[-MAX_TRAIN:]
            current_y = current_y[-MAX_TRAIN:]

        if len(current_X) < MIN_TRAIN:
            return   # finally block still fires → is_training cleared

        X = np.array(current_X)
        y = np.array(current_y)

        if len(np.unique(y)) < 2:
            return   # only 1 class (e.g. 20 frames of "neutral") → finally fires

        skf = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)

        # 3. Clone & fit OUTSIDE every lock — webcam thread never waits
        from sklearn.base import clone
        new_models, new_cv_scores = {}, {}
        for name, pipe in ML_MODELS.items():
            try:
                new_pipe = clone(pipe)
                new_pipe.fit(X, y)
                new_models[name] = new_pipe
                if len(X) >= 10:
                    cv = cross_val_score(new_pipe, X, y, cv=skf, scoring="accuracy")
                    new_cv_scores[name] = cv.tolist()
            except:
                pass

        # 4. Swap trained pipelines under ML lock (microsecond hold)
        with _ml_lock:
            for name, new_pipe in new_models.items():
                ML_MODELS[name] = new_pipe
                _state["ml_trained"][name] = True
            for name, cv in new_cv_scores.items():
                _state["ml_cv_scores"][name] = cv

    finally:
        # 5. GUARANTEED unlock — runs on success, early return, AND exception.
        #    Uses primary _lock because _state is its domain, not _ml_lock.
        with _lock:
            _state["is_training"] = False

# ─────────────────────────────────────────────────────────────────
# BACKGROUND THREADS
# ─────────────────────────────────────────────────────────────────
def webcam_thread():
    """Continuously reads webcam, runs face analysis, updates _state."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()  # FIX #2: ALWAYS drain buffer to prevent lag

        with _lock:
            run_cam = _state["ui_run_webcam"]

        if not run_cam or not ret:
            time.sleep(0.05)
            continue

        face_e = analyze_face(frame)
        with _lock:
            tonal_e = _state["tonal_e"]
            text_e  = _state["text_e"]
            weights = dict(_state["weights"])  # FIX #1: snapshot of live weights
        final_e  = weighted_fusion(face_e, tonal_e, text_e, weights)
        ml_preds = predict_all_models(face_e, tonal_e, text_e)

        # Draw emotion overlay on frame
        annotated = draw_overlay(frame.copy(), face_e, tonal_e, text_e, final_e, ml_preds)

        # Update state
        with _lock:
            _state["frame"]      = annotated
            _state["face_e"]     = face_e
            _state["final_e"]    = final_e
            _state["ml_preds"]   = ml_preds
            _state["frame_count"] += 1
            _state["_fps_n"]     += 1
            if time.time() - _state["_fps_t"] >= 1.0:
                _state["fps"]   = _state["_fps_n"]
                _state["_fps_n"]= 0
                _state["_fps_t"]= time.time()

            # Append session
            t = time.time()
            sess = _state["session"]
            sess["timestamps"].append(t)
            fs   = build_fv(face_e, tonal_e, text_e)
            _state["feature_store"].append(fs)
            for k, v in [("face_seq",face_e),("tonal_seq",tonal_e),
                         ("text_seq",text_e),("final_seq",final_e)]:
                sess[k].append(EMOTION_IDX.get(v, 4))
            for k, v in [("face_str",face_e),("tonal_str",tonal_e),
                         ("text_str",text_e),("final_str",final_e)]:
                sess[k].append(v)

            # FIX #1: Use frame_count (always incrementing) not deque length
            # FIX #3: is_training guard — never spawn overlapping training threads
            fc = _state["frame_count"]
            if (fc % 20 == 0 and fc > 0
                    and len(_state["feature_store"]) >= MIN_TRAIN
                    and not _state["is_training"]):
                _state["is_training"] = True
                threading.Thread(target=train_all_models, daemon=True).start()

        time.sleep(0.05)
    cap.release()


def audio_thread(whisper_model, text_pipeline):
    """Listens to mic, transcribes, extracts tone + text emotion."""
    recognizer = sr.Recognizer()
    mic        = sr.Microphone()
    with _lock:
        _state["audio_active"] = True
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        while True:
            with _lock:
                run_aud = _state["ui_run_audio"]
            if not run_aud:
                time.sleep(0.5)
                continue

            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                audio_np = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32)
                tonal_e  = extract_voice_tonality(audio_np)
                # FIX #5: Use tempfile to avoid multi-session file collision
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                    tmp_f.write(audio.get_wav_data())
                    tmp_name = tmp_f.name
                try:
                    transcript = whisper_model.transcribe(tmp_name, fp16=False)["text"]
                finally:
                    os.remove(tmp_name)
                text_e     = analyze_text(transcript, text_pipeline)
                with _lock:
                    _state["tonal_e"]    = tonal_e
                    _state["text_e"]     = text_e
                    _state["transcript"] = transcript
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                pass
            except Exception as e:
                pass
            time.sleep(0.3)
    except Exception:
        with _lock:
            _state["audio_active"] = False

# ─────────────────────────────────────────────────────────────────
# FRAME OVERLAY  (drawn on webcam frame)
# ─────────────────────────────────────────────────────────────────
def draw_overlay(frame, face_e, tonal_e, text_e, final_e, ml_preds):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_h = 200
    cv2.rectangle(overlay, (0, 0), (380, panel_h), (5, 10, 16), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def ecolor(e):
        mapping = {
            "happy":(45,220,80),"sad":(100,140,255),"angry":(60,60,255),
            "fear":(200,50,200),"disgust":(200,180,30),"surprise":(0,184,255),
            "neutral":(150,150,150)
        }
        return mapping.get(e, (200,200,200))

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"FACE   {face_e.upper()}", face_e, 30),
        (f"VOICE  {tonal_e.upper()}", tonal_e, 58),
        (f"TEXT   {text_e.upper()}", text_e, 86),
    ]
    for label, e, y in lines:
        cv2.putText(frame, label, (12, y), font, 0.52, ecolor(e), 2, cv2.LINE_AA)

    cv2.line(frame, (10, 98), (370, 98), (40,40,40), 1)
    fw_c = ecolor(final_e)
    cv2.putText(frame, f"FUSED  {final_e.upper()}", (12, 122), font, 0.65, fw_c, 2, cv2.LINE_AA)
    cv2.line(frame, (10, 132), (370, 132), (40,40,40), 1)

    y_off = 152
    for nm, pred in list(ml_preds.items())[:4]:
        c = ecolor(pred) if pred != "—" else (60,60,60)
        cv2.putText(frame, f"{nm[:9]:<9} {pred.upper() if pred != '—' else '...'}", 
                    (12, y_off), font, 0.38, c, 1, cv2.LINE_AA)
        y_off += 20

    # FPS badge
    fps = _state.get("fps", 0)
    cv2.putText(frame, f"{fps:.0f} fps", (w-80, 20), font, 0.45, (0,220,120), 1, cv2.LINE_AA)

    # Emotion color bar at bottom
    seg_w = w // len(EMOTIONS)
    for i, e in enumerate(EMOTIONS):
        c = ecolor(e)
        cv2.rectangle(frame, (i*seg_w, h-6), ((i+1)*seg_w, h), c, -1)

    return frame

# ─────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────
def _base_layout(title="", height=260):
    return dict(
        title=dict(text=title, font=dict(color=ACCENT, size=12, family="'Courier New', monospace"),
                   x=0.04, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor =CARD_BG,
        font=dict(color="#c0c0c0", size=10, family="'Courier New', monospace"),
        height=height,
        margin=dict(l=40, r=15, t=40, b=30),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False),
    )

def chart_emotion_timeline(sess):
    final_seq = sess["final_seq"][-120:]
    face_seq  = sess["face_seq"][-120:]
    tonal_seq = sess["tonal_seq"][-120:]

    fig = go.Figure()
    for seq, name, color, dash in [
        (final_seq, "Fused",  ACCENT,  "solid"),
        (face_seq,  "Face",   "#FF6B9D","dot"),
        (tonal_seq, "Voice",  "#FFB800","dashdot"),
    ]:
        if not seq: continue
        fig.add_trace(go.Scatter(
            y=seq, mode="lines",
            name=name,
            line=dict(color=color, width=2 if name=="Fused" else 1.2, dash=dash),
            fill="tozeroy" if name=="Fused" else None,
            fillcolor=f"rgba(0,229,255,0.06)" if name=="Fused" else None,
        ))
    fig.update_layout(**_base_layout("◈ LIVE EMOTION TIMELINE", 220))
    fig.update_yaxes(tickvals=list(range(N_EMOTIONS)),
                     ticktext=[e[:3].upper() for e in EMOTIONS], gridcolor=GRID_COL)
    return fig

def chart_emotion_frequency(sess):
    final_str = sess["final_str"]
    counts = [final_str.count(e) for e in EMOTIONS]
    colors = [EMOTION_COLORS.get(e, "#888") for e in EMOTIONS]
    fig = go.Figure(go.Bar(
        x=[e.upper() for e in EMOTIONS],
        y=counts,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.4)", width=1)),
        text=counts,
        textposition="outside",
        textfont=dict(size=9, color="#aaa"),
    ))
    fig.update_layout(**_base_layout("◈ EMOTION FREQUENCY", 240))
    return fig

def chart_model_accuracy(ml_trained, feature_store, session):
    names, accs = [], []
    if len(feature_store) >= MIN_TRAIN:
        X = np.array(feature_store)
        y = np.array(session["final_seq"])
        if len(np.unique(y)) >= 2:
            with _ml_lock:
                for nm, pipe in ML_MODELS.items():
                    if ml_trained.get(nm):
                        try:
                            acc = accuracy_score(y, pipe.predict(X)) * 100
                            names.append(nm)
                            accs.append(round(acc, 1))
                        except: pass

    if not names:
        fig = go.Figure()
        fig.add_annotation(text=f"Training... (need {MIN_TRAIN} samples)",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#555", size=12), showarrow=False)
        fig.update_layout(**_base_layout("◈ MODEL ACCURACY", 240))
        return fig

    colors = px.colors.sequential.Plasma
    bar_colors = [colors[int(a/100 * (len(colors)-1))] for a in accs]
    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
        textfont=dict(size=10, color="#ccc"),
    ))
    fig.update_layout(**_base_layout("◈ MODEL ACCURACY (TRAIN)", 240))
    fig.update_yaxes(range=[0, 115])
    return fig

def chart_cv_scores(ml_cv_scores):
    trained_cv = {nm: v for nm, v in ml_cv_scores.items() if v}
    if not trained_cv:
        fig = go.Figure()
        fig.add_annotation(text="5-Fold CV — collecting data...",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#555", size=12), showarrow=False)
        fig.update_layout(**_base_layout("◈ 5-FOLD CROSS-VALIDATION", 240))
        return fig

    fig = go.Figure()
    colors_list = [ACCENT, ACCENT2, "#FF6B9D", "#FFB800", "#2DFF7A", "#FF4040"]
    for i, (nm, cv_vals) in enumerate(trained_cv.items()):
        c = colors_list[i % len(colors_list)]
        mean_v = np.mean(cv_vals) * 100
        std_v  = np.std(cv_vals) * 100
        fig.add_trace(go.Bar(
            name=nm,
            x=[nm],
            y=[mean_v],
            error_y=dict(type="data", array=[std_v], visible=True, color="#fff", thickness=1.5),
            marker=dict(color=c, opacity=0.85),
            text=f"{mean_v:.1f}±{std_v:.1f}%",
            textposition="outside",
            textfont=dict(size=9, color="#ccc"),
        ))
    fig.update_layout(**_base_layout("◈ 5-FOLD CV ACCURACY ± STD", 240))
    fig.update_yaxes(range=[0, 115])
    fig.update_layout(showlegend=False, barmode="group")
    return fig

def chart_confusion_matrix(model_name, feature_store, session, ml_trained, colorscale="YlOrRd"):
    if not ml_trained.get(model_name) or len(feature_store) < MIN_TRAIN:
        fig = go.Figure()
        fig.add_annotation(text=f"{model_name} — training...",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#555", size=12), showarrow=False)
        fig.update_layout(**_base_layout(f"◈ {model_name.upper()} CONFUSION", 260))
        return fig

    X = np.array(feature_store)
    y = np.array(session["final_seq"])
    with _ml_lock:
        preds = ML_MODELS[model_name].predict(X)
    cm   = confusion_matrix(y, preds, labels=list(range(N_EMOTIONS)))
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    short = [e[:3].upper() for e in EMOTIONS]

    text_vals = [[f"{cm_n[i][j]:.2f}" for j in range(N_EMOTIONS)] for i in range(N_EMOTIONS)]
    fig = go.Figure(go.Heatmap(
        z=cm_n, x=short, y=short,
        colorscale=colorscale,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=False,
        zmin=0, zmax=1,
    ))
    fig.update_layout(**_base_layout(f"◈ {model_name.upper()} CONFUSION", 260))
    fig.update_yaxes(autorange="reversed")
    return fig

def chart_model_agreement(feature_store, ml_trained):
    trained_names = [nm for nm in ML_MODELS if ml_trained.get(nm)]
    if len(trained_names) < 2 or len(feature_store) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Need ≥2 trained models",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#555", size=12), showarrow=False)
        fig.update_layout(**_base_layout("◈ MODEL AGREEMENT HEATMAP", 260))
        return fig

    X_sub = np.array(feature_store[-50:])
    agree = np.zeros((len(trained_names), len(trained_names)))
    with _ml_lock:
        preds_dict = {}
        for nm in trained_names:
            try:    preds_dict[nm] = ML_MODELS[nm].predict(X_sub)
            except: preds_dict[nm] = np.full(len(X_sub), 4)
    for i, n1 in enumerate(trained_names):
        for j, n2 in enumerate(trained_names):
            agree[i, j] = np.mean(preds_dict[n1] == preds_dict[n2])

    short_names = [nm[:7] for nm in trained_names]
    text_vals   = [[f"{agree[i][j]:.2f}" for j in range(len(trained_names))]
                   for i in range(len(trained_names))]
    fig = go.Figure(go.Heatmap(
        z=agree, x=short_names, y=short_names,
        colorscale="Greens",
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=False,
        zmin=0, zmax=1,
    ))
    fig.update_layout(**_base_layout("◈ MODEL AGREEMENT HEATMAP", 260))
    fig.update_yaxes(autorange="reversed")
    return fig

def chart_rf_feature_importance(ml_trained):
    if not ml_trained.get("RandForest"):
        fig = go.Figure()
        fig.add_annotation(text="RandForest — training...",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#555", size=12), showarrow=False)
        fig.update_layout(**_base_layout("◈ RF FEATURE IMPORTANCE", 240))
        return fig
    with _ml_lock:
        try:
            imp = ML_MODELS["RandForest"].named_steps["clf"].feature_importances_
        except:
            imp = np.zeros(6)
    feat_names = ["Face", "Tonal", "Text", "F-conf", "T-conf", "X-conf"]
    colors_ = [EMOTION_COLORS.get("happy","#2DFF7A"),
               EMOTION_COLORS.get("sad","#5B8CFF"),
               EMOTION_COLORS.get("surprise","#FFB800"),
               EMOTION_COLORS.get("fear","#C832C8"),
               EMOTION_COLORS.get("angry","#FF3B3B"),
               EMOTION_COLORS.get("disgust","#20D4D4")]
    fig = go.Figure(go.Bar(
        x=imp, y=feat_names, orientation="h",
        marker=dict(color=colors_, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        text=[f"{v:.3f}" for v in imp],
        textposition="outside",
        textfont=dict(size=9, color="#ccc"),
    ))
    fig.update_layout(**_base_layout("◈ RF FEATURE IMPORTANCE", 240))
    fig.update_xaxes(range=[0, max(imp)*1.25 + 0.01])
    return fig

def chart_emotion_donut(sess):
    final_str = sess["final_str"]
    counts = [final_str.count(e) for e in EMOTIONS]
    total  = sum(counts)
    if total == 0: counts = [1]*N_EMOTIONS
    colors_ = [EMOTION_COLORS.get(e, "#888") for e in EMOTIONS]
    fig = go.Figure(go.Pie(
        labels=[e.upper() for e in EMOTIONS],
        values=counts,
        hole=0.6,
        marker=dict(colors=colors_, line=dict(color=CARD_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=9),
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10,r=10,t=30,b=10),
        title=dict(text="◈ DISTRIBUTION", font=dict(color=ACCENT, size=11,
                   family="'Courier New', monospace"), x=0.04, xanchor="left"),
        font=dict(color="#c0c0c0", family="'Courier New', monospace"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# STREAMLIT PAGE SETUP
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment AI — Live Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

  html, body, [class*="css"] {{
    background-color: {BG} !important;
    color: #c0c0c0 !important;
    font-family: 'Share Tech Mono', monospace !important;
  }}
  .block-container {{ padding: 1rem 1.5rem 1rem 1.5rem !important; max-width:100% !important; }}
  
  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: #080f15 !important;
    border-right: 1px solid rgba(0,229,255,0.15);
  }}
  section[data-testid="stSidebar"] * {{ color: #9ab !important; font-family: 'Share Tech Mono', monospace !important; }}
  
  /* Metric cards */
  [data-testid="metric-container"] {{
    background: {CARD_BG} !important;
    border: 1px solid rgba(0,229,255,0.18) !important;
    border-radius: 4px !important;
    padding: 0.5rem 0.8rem !important;
  }}
  [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-family: 'Orbitron', monospace !important; font-size:1.1rem !important; }}
  [data-testid="stMetricLabel"] {{ color: #567 !important; font-size: 0.68rem !important; letter-spacing:0.1em; }}
  [data-testid="stMetricDelta"] {{ font-size: 0.72rem !important; }}

  /* Header */
  .dash-header {{
    font-family: 'Orbitron', monospace;
    font-size: 1.55rem;
    font-weight: 900;
    color: {ACCENT};
    letter-spacing: 0.12em;
    text-shadow: 0 0 18px rgba(0,229,255,0.5);
    line-height: 1.1;
  }}
  .dash-sub {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #345;
    letter-spacing: 0.2em;
  }}
  
  /* Emotion badge */
  .emo-badge {{
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 3px;
    font-family: 'Orbitron', monospace;
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    border: 1px solid currentColor;
    text-transform: uppercase;
  }}
  
  /* Card */
  .info-card {{
    background: {CARD_BG};
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
  }}
  
  /* Live dot */
  .live-dot {{
    display:inline-block;
    width:9px; height:9px;
    border-radius:50%;
    background: #2DFF7A;
    box-shadow: 0 0 8px #2DFF7A;
    animation: blink 1.2s ease infinite;
    margin-right: 6px;
  }}
  @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.2}} }}
  
  /* Transcript */
  .transcript-box {{
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.15);
    border-left: 3px solid {ACCENT};
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: #8ab;
    font-style: italic;
    border-radius: 0 4px 4px 0;
    min-height: 40px;
  }}
  
  /* Divider */
  .scan-line {{
    border: none;
    border-top: 1px solid rgba(0,229,255,0.12);
    margin: 0.6rem 0;
  }}
  
  /* Kappa row */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 0.4rem;
  }}
  .stat-item {{
    background:{CARD_BG};
    border:1px solid rgba(0,229,255,0.1);
    border-radius:3px;
    padding:0.4rem 0.6rem;
    font-size:0.7rem;
    text-align:center;
  }}
  .stat-val {{ color:{ACCENT}; font-size:1.0rem; font-family:'Orbitron',monospace; display:block; margin-top:3px; }}
  
  /* Scrollbar */
  ::-webkit-scrollbar {{ width:4px; }}
  ::-webkit-scrollbar-track {{ background:{BG}; }}
  ::-webkit-scrollbar-thumb {{ background:rgba(0,229,255,0.25); border-radius:2px; }}
  
  /* Plotly container spacing */
  .element-container {{ margin-bottom: 0 !important; padding-bottom: 0 !important; }}
  
  /* Status bar */
  .status-bar {{
    display:flex; gap:1rem; align-items:center;
    padding: 0.4rem 0.8rem;
    background: #080f15;
    border-radius:3px;
    border: 1px solid rgba(0,229,255,0.08);
    font-size: 0.7rem; color: #456;
    margin-bottom: 0.6rem;
  }}
  .status-ok {{ color: #2DFF7A; }}
  .status-warn {{ color: #FFB800; }}
  
  /* Override streamlit blue focus */
  button[kind="primary"] {{ background: {ACCENT} !important; color: #000 !important; font-family:'Orbitron',monospace !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="dash-header" style="font-size:1rem">⬡ CONTROL CENTER</p>', unsafe_allow_html=True)
    st.markdown('<hr class="scan-line">', unsafe_allow_html=True)

    run_audio = st.toggle("🎙 Voice Analysis", value=True)
    show_webcam= st.toggle("📷 Webcam Feed", value=True)

    st.markdown("**Fusion Weights**")
    fw_face   = st.slider("Face weight",  0, 100, int(FACE_WEIGHT),  step=5)
    fw_tonal  = st.slider("Voice weight", 0, 100, int(TONAL_WEIGHT), step=5)
    fw_text   = st.slider("Text weight",  0, 100, int(TEXT_WEIGHT),  step=5)
    FACE_WEIGHT  = float(fw_face)
    TONAL_WEIGHT = float(fw_tonal)
    TEXT_WEIGHT  = float(fw_text)

    st.markdown("**Session**")
    refresh_ms = st.slider("Refresh rate (ms)", 200, 2000, 500, step=100)

    st.markdown('<hr class="scan-line">', unsafe_allow_html=True)
    n_samples  = len(_state["session"]["final_seq"])
    n_trained  = sum(_state["ml_trained"].values())
    st.markdown(f"""
    <div class="info-card" style="font-size:0.72rem">
      <span style="color:#567">SAMPLES</span>
      <span style="color:{ACCENT};float:right;font-family:'Orbitron',monospace">{n_samples}</span><br>
      <span style="color:#567">MODELS TRAINED</span>
      <span style="color:{'#2DFF7A' if n_trained==6 else '#FFB800'};float:right">{n_trained}/6</span><br>
      <span style="color:#567">MIN FOR TRAIN</span>
      <span style="color:#456;float:right">{MIN_TRAIN}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**About**")
    st.markdown("""
    <div style="font-size:0.65rem;color:#345;line-height:1.7">
    ◈ DeepFace facial emotion<br>
    ◈ Librosa voice tonality<br>
    ◈ RoBERTa text sentiment<br>
    ◈ Whisper ASR transcription<br>
    ◈ 6 ML classifiers live<br>
    ◈ Cohen's κ · Pearson r · Entropy
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# START BACKGROUND THREADS
# FIX #2: No arguments on cache_resource fn — UI toggle never re-triggers it
# FIX #3 (prev): cache_resource = true app-level singleton, survives tab refresh
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def start_hardware_threads():
    threading.Thread(target=webcam_thread, daemon=True).start()
    # Always start audio — use _state["audio_active"] flag inside thread to pause
    wm = load_whisper()
    tp = load_text_pipeline()
    threading.Thread(target=audio_thread, args=(wm, tp), daemon=True).start()
    return True

start_hardware_threads()   # FIX #2: unconditional, argument-free

# Sync UI toggles → background threads on every render pass
with _lock:
    _state["ui_run_audio"]  = run_audio
    _state["ui_run_webcam"] = show_webcam
    _state["weights"]["face"]  = float(fw_face)
    _state["weights"]["tonal"] = float(fw_tonal)
    _state["weights"]["text"]  = float(fw_text)

# ─────────────────────────────────────────────────────────────────
# READ CURRENT STATE  (snapshot for this render pass)
# ─────────────────────────────────────────────────────────────────
with _lock:
    frame        = _state["frame"]
    face_e       = _state["face_e"]
    tonal_e      = _state["tonal_e"]
    text_e       = _state["text_e"]
    final_e      = _state["final_e"]
    ml_preds     = dict(_state["ml_preds"])
    transcript   = _state["transcript"]
    session      = {k: list(v) for k, v in _state["session"].items()}
    feature_store= list(_state["feature_store"])
    ml_trained   = dict(_state["ml_trained"])
    ml_cv_scores = {k: list(v) for k, v in _state["ml_cv_scores"].items()}
    fps_val      = _state["fps"]
    audio_active = _state["audio_active"]

n_samples = len(session["final_seq"])

# ─────────────────────────────────────────────────────────────────
# HEADER ROW
# ─────────────────────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3,1])
with h_col1:
    st.markdown("""
    <p class="dash-header">⬡ MULTIMODAL SENTIMENT INTELLIGENCE</p>
    <p class="dash-sub">FACE · VOICE · TEXT · ML ENSEMBLE · LIVE ANALYTICS</p>
    """, unsafe_allow_html=True)
with h_col2:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align:right;padding-top:0.3rem">
      <span class="live-dot"></span>
      <span style="color:#2DFF7A;font-family:'Orbitron',monospace;font-size:0.85rem">LIVE</span><br>
      <span style="color:#456;font-size:0.7rem">{ts}</span><br>
      <span style="color:#345;font-size:0.65rem">n={n_samples} · {fps_val:.0f}fps</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="scan-line">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────────────────────────
n_tr = sum(ml_trained.values())
aud_status = "ACTIVE" if audio_active else "STANDBY"
aud_cls    = "status-ok" if audio_active else "status-warn"
st.markdown(f"""
<div class="status-bar">
  <span>WEBCAM <span class="status-ok">▶ CAPTURING</span></span>
  <span>AUDIO <span class="{aud_cls}">▶ {aud_status}</span></span>
  <span>MODELS <span class="{'status-ok' if n_tr==6 else 'status-warn'}">{n_tr}/6 TRAINED</span></span>
  <span>SAMPLES <span style="color:#7ab">{n_samples}</span></span>
  <span>FPS <span style="color:#7ab">{fps_val:.0f}</span></span>
  <span style="margin-left:auto;color:#234">FUSION: FACE {int(FACE_WEIGHT)}% · VOICE {int(TONAL_WEIGHT)}% · TEXT {int(TEXT_WEIGHT)}%</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# ROW 1: WEBCAM + CURRENT EMOTIONS + ML PREDICTIONS
# ─────────────────────────────────────────────────────────────────
cam_col, emo_col, ml_col = st.columns([2, 1.4, 1.6])

with cam_col:
    st.markdown('<div style="font-size:0.7rem;color:#345;letter-spacing:0.15em;margin-bottom:4px">◈ LIVE FEED</div>', unsafe_allow_html=True)
    if show_webcam and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
    elif frame is None:
        st.markdown("""
        <div style="background:#0d1620;border:1px solid rgba(0,229,255,0.1);
             border-radius:4px;height:240px;display:flex;align-items:center;
             justify-content:center;color:#234;font-size:0.8rem">
          ◈ AWAITING WEBCAM...
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#345;font-size:0.75rem;padding:2rem;text-align:center">Feed hidden</div>', unsafe_allow_html=True)

with emo_col:
    st.markdown('<div style="font-size:0.7rem;color:#345;letter-spacing:0.15em;margin-bottom:6px">◈ CURRENT READING</div>', unsafe_allow_html=True)

    def emo_card(label, emotion, icon=""):
        c = EMOTION_COLORS.get(emotion, "#888")
        return f"""
        <div style="background:{CARD_BG};border:1px solid {c}33;border-left:3px solid {c};
             border-radius:3px;padding:0.45rem 0.7rem;margin-bottom:0.35rem;display:flex;
             justify-content:space-between;align-items:center">
          <span style="font-size:0.65rem;color:#456;letter-spacing:0.1em">{label}</span>
          <span style="color:{c};font-family:'Orbitron',monospace;font-size:0.8rem;font-weight:700">{icon} {emotion.upper()}</span>
        </div>"""

    st.markdown(
        emo_card("FACE EXPRESSION", face_e,  "◈") +
        emo_card("VOICE TONALITY",  tonal_e, "◈") +
        emo_card("SPEECH TEXT",     text_e,  "◈"),
        unsafe_allow_html=True
    )

    final_c = EMOTION_COLORS.get(final_e, "#888")
    st.markdown(f"""
    <div style="background:{CARD_BG};border:2px solid {final_c}88;border-radius:4px;
         padding:0.55rem 0.8rem;text-align:center;margin-top:0.3rem">
      <div style="font-size:0.62rem;color:#456;letter-spacing:0.18em;margin-bottom:4px">⬡ WEIGHTED FUSION</div>
      <div style="color:{final_c};font-family:'Orbitron',monospace;font-size:1.35rem;
           font-weight:900;text-shadow:0 0 12px {final_c}66">{final_e.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Transcript
    short_t = transcript[-90:] if len(transcript) > 90 else transcript
    st.markdown(f'<div class="transcript-box">"{short_t or "— listening —"}"</div>', unsafe_allow_html=True)

with ml_col:
    st.markdown('<div style="font-size:0.7rem;color:#345;letter-spacing:0.15em;margin-bottom:6px">◈ ML MODEL PREDICTIONS</div>', unsafe_allow_html=True)
    model_colors_list = [ACCENT, ACCENT2, "#FF6B9D", "#FFB800", "#2DFF7A", "#FF4040"]
    for i, (nm, pred) in enumerate(ml_preds.items()):
        c       = EMOTION_COLORS.get(pred, "#345") if pred != "—" else "#234"
        mc      = model_colors_list[i % len(model_colors_list)]
        trained = ml_trained.get(nm, False)
        tag     = "TRAINED" if trained else f"need {MIN_TRAIN}"
        tag_c   = "#2DFF7A" if trained else "#FFB800"
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid rgba(255,255,255,0.05);
             border-radius:3px;padding:0.35rem 0.7rem;margin-bottom:0.28rem;
             display:flex;justify-content:space-between;align-items:center">
          <span style="color:{mc};font-size:0.68rem;letter-spacing:0.05em">{nm}</span>
          <span style="color:#234;font-size:0.58rem">({tag_c and f'<span style="color:{tag_c}">{tag}</span>' })</span>
          <span style="color:{c};font-family:'Orbitron',monospace;font-size:0.75rem;font-weight:700">{pred.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    # Donut chart
    st.plotly_chart(chart_emotion_donut(session), use_container_width=True, config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────────
# ROW 2: EMOTION TIMELINE + FREQUENCY
# ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="scan-line">', unsafe_allow_html=True)
tl_col, fr_col = st.columns([2,1])
with tl_col:
    st.plotly_chart(chart_emotion_timeline(session), use_container_width=True, config={"displayModeBar":False})
with fr_col:
    st.plotly_chart(chart_emotion_frequency(session), use_container_width=True, config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────────
# ROW 3: ML MODEL COMPARISON CHARTS
# ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="scan-line">', unsafe_allow_html=True)
acc_col, cv_col, fi_col = st.columns(3)
with acc_col:
    st.plotly_chart(chart_model_accuracy(ml_trained, feature_store, session),
                    use_container_width=True, config={"displayModeBar":False})
with cv_col:
    st.plotly_chart(chart_cv_scores(ml_cv_scores),
                    use_container_width=True, config={"displayModeBar":False})
with fi_col:
    st.plotly_chart(chart_rf_feature_importance(ml_trained),
                    use_container_width=True, config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────────
# ROW 4: CONFUSION MATRICES + AGREEMENT HEATMAP
# ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="scan-line">', unsafe_allow_html=True)
cm1_col, cm2_col, cm3_col, agr_col = st.columns(4)
with cm1_col:
    st.plotly_chart(chart_confusion_matrix("LogReg",     feature_store, session, ml_trained, "YlOrRd"),
                    use_container_width=True, config={"displayModeBar":False})
with cm2_col:
    st.plotly_chart(chart_confusion_matrix("SVM",        feature_store, session, ml_trained, "Blues"),
                    use_container_width=True, config={"displayModeBar":False})
with cm3_col:
    st.plotly_chart(chart_confusion_matrix("RandForest", feature_store, session, ml_trained, "Greens"),
                    use_container_width=True, config={"displayModeBar":False})
with agr_col:
    st.plotly_chart(chart_model_agreement(feature_store, ml_trained),
                    use_container_width=True, config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────────
# ROW 5: STATISTICAL METRICS
# ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="scan-line">', unsafe_allow_html=True)
st.markdown('<div style="font-size:0.7rem;color:#345;letter-spacing:0.15em;margin-bottom:8px">◈ STATISTICAL SUMMARY</div>', unsafe_allow_html=True)

ent      = emotion_entropy(session["final_seq"])
max_ent  = np.log2(N_EMOTIONS)
kap_ft   = kappa(session["face_seq"],  session["tonal_seq"])
kap_fx   = kappa(session["face_seq"],  session["text_seq"])
kap_tx   = kappa(session["tonal_seq"], session["text_seq"])
cor_ft   = modality_corr(session["face_seq"],  session["tonal_seq"])
cor_fx   = modality_corr(session["face_seq"],  session["text_seq"])
cor_tx   = modality_corr(session["tonal_seq"], session["text_seq"])

m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
m1.metric("Shannon Entropy",  f"{ent:.3f}",     f"/{max_ent:.2f} max")
m2.metric("κ Face↔Voice",     f"{kap_ft:.3f}",  "Cohen's Kappa")
m3.metric("κ Face↔Text",      f"{kap_fx:.3f}",  "Cohen's Kappa")
m4.metric("κ Voice↔Text",     f"{kap_tx:.3f}",  "Cohen's Kappa")
m5.metric("r Face↔Voice",     f"{cor_ft:.3f}",  "Pearson r")
m6.metric("r Face↔Text",      f"{cor_fx:.3f}",  "Pearson r")
m7.metric("r Voice↔Text",     f"{cor_tx:.3f}",  "Pearson r")
m8.metric("Models Ready",     f"{n_tr}/6",       "trained")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:1.5rem;padding:0.6rem 1rem;background:#050a0e;
     border-top:1px solid rgba(0,229,255,0.08);display:flex;
     justify-content:space-between;font-size:0.62rem;color:#234">
  <span>⬡ MULTIMODAL SENTIMENT INTELLIGENCE · DeepFace + Librosa + RoBERTa + Whisper</span>
  <span>6 ML CLASSIFIERS · LogReg · RandForest · SVM · MLP · KNN · NaiveBayes</span>
  <span>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────
time.sleep(refresh_ms / 1000.0)
st.rerun()
