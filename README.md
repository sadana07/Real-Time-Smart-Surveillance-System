# CCTV Security AI

Detects what's in front of a camera:
- **Person** → predicts gender + approximate age
- **Animal** (dog, cat, cow, bird, etc.) → predicts species name
- **Insect** → predicts species name (stub, needs training — see below)

## Windows Setup (copy-paste)

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

Press `q` to quit the video window.

## What's real vs. stub right now

| Component | Status |
|---|---|
| Person detection | Working (YOLOv8, pretrained) |
| Gender prediction | Working (DeepFace, pretrained) |
| Animal name (dog/cat/cow/bird/etc.) | Working (YOLOv8 COCO classes) |

## Next steps to make it resume-ready

1. **Fine-tune YOLOv8 on Indian animals**: pull an open dataset from Roboflow
   (search "Indian wildlife" or "monkey/cow/snake detection"), fine-tune
   `yolov8n.pt` for ~50 epochs.
2. **Train the insect classifier**: download IP102 from Kaggle, fine-tune
   a MobileNetV2 on it, save as `insect_model.h5`, drop it in this folder —
   `insect_classifier.py` will pick it up automatically.
3. **Wrap in FastAPI + React dashboard** (like your Crop Yield project) so it
   can run as a live-feed web app instead of a local OpenCV window — good
   for demoing on LinkedIn/resume.
4. **Add a logging layer**: write detections (timestamp, label) to a CSV or
   SQLite DB, so you can show a "detection history" table in the dashboard.
5. **Limitations to mention in your report**: age/gender accuracy drops with
   poor lighting/angle (typical CCTV footage); insect detection needs a
   dedicated small-object detector since COCO has zero insect classes.

## Known accuracy caveats (mention these in your project report — shows maturity)

- Age prediction from a single face crop is approximate (±5-8 years typical error).
- Gender prediction accuracy drops on partial/occluded faces — common in real CCTV angles.
- Nighttime/IR camera footage will need a model retrained on IR data; visible-light
  models (like the ones here) perform poorly in the dark.
