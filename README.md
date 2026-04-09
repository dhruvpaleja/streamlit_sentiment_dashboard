# Multimodal Sentiment Intelligence — Live Streamlit Dashboard

Real-time multimodal sentiment analysis combining **Face + Voice + Text** with **6 live ML classifiers** and interactive Plotly analytics.

## Features

- **DeepFace** facial emotion recognition (7 emotions)
- **Librosa** voice tonality analysis
- **RoBERTa** (Hugging Face) text sentiment analysis
- **Whisper** speech-to-text transcription
- **6 ML classifiers** trained live: LogReg, RandForest, SVM, MLP, KNN, NaiveBayes
- **Live Plotly charts**: timeline, frequency bars, confusion matrices, agreement heatmap, RF feature importance, donut distribution
- **Statistical metrics**: Shannon entropy, Cohen's κ, Pearson r
- **Dark cyberpunk UI** with custom CSS

## Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_sentiment_dashboard.py
```

## Architecture

- Background threads handle webcam and audio continuously
- Shared `_state` dictionary with thread-safe locking
- Streamlit auto-refreshes the UI on each render pass
- ML models retrain every 20 frames with a concurrency guard

## Deployment

Deploy on [Streamlit Community Cloud](https://share.streamlit.io/):
1. Push this repo to GitHub
2. Connect it at `share.streamlit.io`
3. Select the repo, branch, and main file
