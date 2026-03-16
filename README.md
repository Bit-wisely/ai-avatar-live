# Real-Time AI Avatar System

A production-ready pipeline for real-time facial animation and voice conversion.

## Features
- **Webcam Tracking**: MediaPipe for high-speed face landmark detection.
- **AI Animation**: LivePortrait implementation for static image animation.
- **Voice Conversion**: RVC (Retrieval-based Voice Conversion) for real-time cloning.
- **Multithreaded Pipeline**: Low-latency execution (< 200ms).

## Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and in System PATH.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the setup script to download models:
   ```bash
   bash setup.sh
   # or
   python scripts/download_models.py
   ```

## Model Weights
- **LivePortrait**: Download weights into `models/liveportrait/`.
- **RVC**: Download base models and voice indexes into `models/rvc_voice/`.

## Running the System
```bash
python run.py
```

## Repository Structure
- `src/camera/`: Handles webcam I/O.
- `src/tracking/`: MediaPipe facial landmark logic.
- `src/animation/`: LivePortrait renderer.
- `src/audio/`: Microphone and RVC processing.
- `src/pipeline/`: Thread management and synchronization.
- `src/streaming/`: Output rendering and WebSockets.
