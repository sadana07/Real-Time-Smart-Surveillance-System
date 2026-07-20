# Real Time Smart Surveillance System

Detects what's in front of a camera:
- **Person** → predicts gender 
- **Animal** (dog, cat, cow, bird, etc.) → predicts species name
- **Vehicles** (car, motorcycle, truck, bus, etc.) → predicts vehicle common name


## Windows Setup

```powershell
# 1. Create venv (use Python 3.11 — same as your MediaPipe projects)
py -3.11 -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run on webcam
python detector.py --source 0

# 4. Or run on a video file
python detector.py --source path\to\video.mp4
```

Press "q" to quit the video window.

## What's real vs. stub right now

| Component | Status |
|---|---|
| Person detection | Working (YOLOv8) |
| Gender prediction | Working (DeepFace) |
| Animal name (dog/cat/cow/bird/etc.) | Working (YOLOv8 COCO classes) |

## Why it is useful
- Real-time monitoring of people and vehicles.
- Reduces the need for continuous manual CCTV observation.
- Provides instant visual identification and analytics.
- Can be extended with features like face recognition, intrusion detection, crowd analysis, or suspicious activity detection.

