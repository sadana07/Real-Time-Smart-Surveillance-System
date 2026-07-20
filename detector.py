"""
detector.py
CCTV Security AI - Person (gender) + Vehicle detector

Usage:
    python detector.py --source 0                # webcam
    python detector.py --source path/to/video.mp4 # video file

Pipeline:
    1. YOLOv8 (COCO-pretrained) detects every object in the frame, so
       multiple people/vehicles/animals are all handled every frame — no
       special-casing needed for "multiple vehicles", it just works
       because every detected box gets its own label independently.
    2. class == "person"      -> crop face -> predict gender (Male/Female)
    3. class in KNOWN_VEHICLES -> label directly with the vehicle name
       (e.g. "Vehicle: car", "Vehicle: truck") — every vehicle box found
       in the frame gets drawn and labeled, so if multiple vehicles are
       spotted, each one shows its own name.
    4. class in KNOWN_ANIMALS  -> label directly with the animal name
       (e.g. "Animal: cow", "Animal: dog", "Animal: Sheep/Goat"). Tuned
       for common Indian street/CCTV scenes: cow, dog, cat, bird, sheep.

NOTE on crows and goats specifically:
    COCO (what YOLOv8 is pretrained on) has NO "crow" class and NO "goat"
    class — only generic "bird" and "sheep" classes. So a crow is
    labeled "Animal: bird", and a goat is labeled "Animal: Sheep/Goat"
    (since a goat gets detected as the closest built-in class, "sheep",
    and we display that honestly rather than claiming false certainty).
    Distinguishing crow-vs-other-bird or goat-vs-sheep needs a
    custom-trained model on labeled images; it is NOT something the
    stock weights can do, regardless of confidence threshold or distance
    tuning.

NOTE on distant/small detections:
    yolov8n (nano) trades accuracy for speed, so small/far-away objects
    (a cow or dog that's a small speck in a wide CCTV frame) are the
    hardest case for it and may simply not be detected at all. Two knobs
    below help with this:
      - ANIMAL_CONF_THRESHOLD: lets animals use a lower confidence
        cutoff than people/vehicles, so faint/distant detections aren't
        thrown away.
      - YOLO_MODEL: switch to "yolov8s.pt" or "yolov8m.pt" for
        meaningfully better small-object recall (slower, needs more
        compute — fine on GPU, noticeably slower on CPU-only webcam use).
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ---- Config ----------------------------------------------------------
# nano = fastest, good for CPU/webcam but weakest on small/distant objects.
# For better recall on distant cows/dogs/birds, switch to "yolov8s.pt"
# (better accuracy, still workable speed) or "yolov8m.pt" (best accuracy,
# needs a decent GPU to stay real-time).
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.45            # default threshold, used for person/vehicle

# Animals (especially far away in a wide CCTV frame) are small and easy to
# miss at the default threshold, so they get their own lower cutoff. Lower
# = more detections but more false positives; raise this back up if you
# start seeing animals detected where there are none.
ANIMAL_CONF_THRESHOLD = 0.30

# If True, the frame is upscaled before running YOLO (then boxes are
# scaled back down to original frame size). This gives small/distant
# objects more pixels to be detected from, at the cost of slower
# inference. Useful for wide-angle CCTV shots where animals are far away.
UPSCALE_FOR_DISTANT_DETECTION = True
UPSCALE_FACTOR = 1.5              # try 1.5-2.0; higher = slower but catches smaller objects

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

# A real face is roughly as wide as it is tall. Haar face-cascade false
# positives on hands/fingers/knuckles tend to produce oddly shaped boxes
# (too narrow, too wide, or too small relative to the frame), so we
# reject anything outside this ratio before trusting it's a face.
FACE_MIN_ASPECT_RATIO = 0.65   # width / height
FACE_MAX_ASPECT_RATIO = 1.45
FACE_MIN_SIZE_PX = 40          # ignore tiny "face" matches (likely noise)

# After a face-shaped region passes the check above, require at least
# this many eye-like features inside it (a hand/finger has none; a real
# face has two) before accepting it and running gender prediction.
MIN_EYES_REQUIRED = 1

# Gender model files
# Preferred: your own trained ONNX model (from train_age_gender_UTKFace.py —
#            only the gender_model.onnx output is used now)
# Fallback: the original pretrained Caffe model (download_models.py)
GENDER_ONNX = "gender_model.onnx"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ["Male", "Female"]
ONNX_INPUT_SIZE = 96  # must match IMG_SIZE in train_age_gender_UTKFace.py

# COCO class names that map to vehicles — already built into YOLO/COCO,
# no extra training needed for this part.
KNOWN_VEHICLES = {
    "bicycle", "car", "motorcycle", "bus", "truck", "train", "airplane", "boat"
}

# COCO class names that map to animals common in Indian street/CCTV
# scenes. Note: "bird" also covers crows (see module docstring) since
# COCO has no dedicated crow class. Likewise COCO has no "goat" class —
# goats get detected as "sheep" (the closest built-in class), so we
# display that as "Sheep/Goat" rather than falsely claiming certainty.
KNOWN_ANIMALS = {
    "cow", "dog", "cat", "bird", "sheep"
}

# Overrides for how a raw COCO class name is displayed as a label.
# Any class not listed here just displays as its own name (title-cased).
ANIMAL_DISPLAY_NAMES = {
    "sheep": "Sheep/Goat",
}
# ------------------------------------------------------------------------


def load_models():
    print("[INFO] Loading YOLOv8 model...")
    yolo = YOLO(YOLO_MODEL)

    print("[INFO] Loading face cascade...")
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    print("[INFO] Loading eye cascade (used to reject hand/finger false positives)...")
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    print("[INFO] Loading gender model...")
    use_onnx = os.path.exists(GENDER_ONNX)

    if use_onnx:
        print("[INFO] Found your own trained gender_model.onnx — using it.")
        gender_net = cv2.dnn.readNetFromONNX(GENDER_ONNX)
    else:
        print("[INFO] No trained ONNX model found — using pretrained Caffe model.")
        try:
            gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        except cv2.error:
            print("[ERROR] Gender model files not found.")
            print("        Run this first:  python download_models.py")
            raise SystemExit(1)

    return yolo, face_cascade, eye_cascade, gender_net, use_onnx


def predict_gender(face_crop_bgr, gender_net, use_onnx):
    """
    Runs gender inference on a cropped face image.
    Returns gender:str ("Male"/"Female") or None if it fails.
    """
    try:
        if use_onnx:
            img = cv2.resize(face_crop_bgr, (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            blob = img.reshape(1, ONNX_INPUT_SIZE, ONNX_INPUT_SIZE, 3)
        else:
            blob = cv2.dnn.blobFromImage(
                face_crop_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
            )

        gender_net.setInput(blob)
        preds = gender_net.forward()
        return GENDER_LIST[preds[0].argmax()]
    except Exception:
        import traceback
        print("[WARN] Gender prediction failed. Full error below:")
        traceback.print_exc()
        return None


def process_person(frame, box, face_cascade, eye_cascade, gender_net, use_onnx):
    """
    Handles ONE detected person box. Called once per person, so multiple
    people in frame each get their own independent gender label.

    Guards against Haar face-cascade false positives on hands/fingers:
    skin texture, knuckle creases, and shadows on a raised hand can
    accidentally pattern-match as a "face". Two checks filter these out
    before gender prediction runs:
      1. Shape check   - a real face is roughly square; reject boxes
                          that are too narrow/wide or too small.
      2. Eye check     - require at least one eye-like feature inside
                          the candidate face region. A hand/finger has
                          none; a real face does.
    Only a region that survives both is treated as an actual face.
    """
    x1, y1, x2, y2 = box
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return "Person"

    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8,
                                           minSize=(FACE_MIN_SIZE_PX, FACE_MIN_SIZE_PX))

    for (fx, fy, fw, fh) in faces:
        aspect_ratio = fw / float(fh)
        if not (FACE_MIN_ASPECT_RATIO <= aspect_ratio <= FACE_MAX_ASPECT_RATIO):
            continue  # wrong shape to be a real face -> likely a hand/finger

        face_gray = gray[fy:fy + fh, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6)
        if len(eyes) < MIN_EYES_REQUIRED:
            continue  # no eyes found -> not a real face

        face_crop = person_crop[fy:fy + fh, fx:fx + fw]
        if face_crop.size == 0:
            continue

        gender = predict_gender(face_crop, gender_net, use_onnx)
        if gender is None:
            return "Person (gender unclear)"
        return gender

    # No candidate region passed both checks -> either no face is visible,
    # or what the cascade matched was a false positive (e.g. a raised
    # hand/finger), so we deliberately do NOT guess.
    return "Person (face not visible)"


def process_frame(frame, yolo, face_cascade, eye_cascade, gender_net, use_onnx):
    orig_h, orig_w = frame.shape[:2]

    # Upscale before inference so small/distant objects have more pixels
    # to be detected from, then scale detected boxes back down to the
    # original frame size for drawing.
    if UPSCALE_FOR_DISTANT_DETECTION:
        infer_frame = cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                                  interpolation=cv2.INTER_LINEAR)
        scale_back = 1.0 / UPSCALE_FACTOR
    else:
        infer_frame = frame
        scale_back = 1.0

    # Run once at the lower of the two thresholds so nothing is dropped
    # before we get a chance to apply the per-class threshold ourselves.
    run_conf = min(CONF_THRESHOLD, ANIMAL_CONF_THRESHOLD)
    results = yolo(infer_frame, conf=run_conf, verbose=False)[0]
    labels_drawn = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo.names[cls_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if scale_back != 1.0:
            x1, y1, x2, y2 = (int(round(v * scale_back)) for v in (x1, y1, x2, y2))
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        if cls_name == "person":
            if confidence < CONF_THRESHOLD:
                continue
            label = process_person(frame, (x1, y1, x2, y2), face_cascade, eye_cascade, gender_net, use_onnx)
            color = (0, 255, 0)
        elif cls_name in KNOWN_VEHICLES:
            if confidence < CONF_THRESHOLD:
                continue
            # every vehicle box gets its own name, so multiple vehicles
            # in frame are each labeled independently (e.g. "Vehicle: car",
            # "Vehicle: truck")
            label = f"Vehicle: {cls_name}"
            color = (0, 140, 255)
        elif cls_name in KNOWN_ANIMALS:
            if confidence < ANIMAL_CONF_THRESHOLD:
                continue
            # "bird" also covers crows, "sheep" also covers goats —
            # COCO has no separate crow or goat class
            display_name = ANIMAL_DISPLAY_NAMES.get(cls_name, cls_name)
            label = f"Animal: {display_name}"
            color = (255, 140, 0)
        else:
            # not a person, known vehicle, or known animal -> skip
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        labels_drawn.append(label)

    return frame, labels_drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="0 for webcam, or path to a video file")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    yolo, face_cascade, eye_cascade, gender_net, use_onnx = load_models()
    print("[INFO] Starting detection loop. Press 'q' to quit.")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame, labels = process_frame(frame, yolo, face_cascade, eye_cascade, gender_net, use_onnx)

        # FPS overlay
        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("CCTV Security AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()