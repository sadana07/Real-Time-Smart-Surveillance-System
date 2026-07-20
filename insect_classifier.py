"""
insect_classifier.py

Placeholder insect classifier.
COCO (what YOLOv8 is pretrained on) has NO insect classes, so YOLO alone
cannot name insects. To make this real:

1. Download the IP102 dataset (102 insect species, Kaggle: "IP102").
2. Fine-tune a lightweight classifier (MobileNetV2 / EfficientNet-B0)
   on cropped insect images.
3. Save the trained model as insect_model.h5 (or .pt if using PyTorch)
   in this folder.
4. Replace the body of classify_insect() below with real inference.

Until step 2-4 are done, this module returns "Unidentified Insect" so the
rest of the pipeline (detector.py) still runs end-to-end.
"""

import numpy as np

INSECT_MODEL_PATH = "insect_model.h5"
_model = None


def _load_model():
    global _model
    if _model is None:
        try:
            from tensorflow.keras.models import load_model
            _model = load_model(INSECT_MODEL_PATH)
        except Exception:
            _model = False  # mark as "tried and failed" so we don't retry every frame
    return _model


def classify_insect(cropped_bgr_image: np.ndarray) -> str:
    """
    Takes a cropped BGR image (numpy array) containing a suspected insect
    and returns a species name string.

    Currently a stub — returns a generic label until a trained model is
    plugged in (see module docstring for training steps).
    """
    model = _load_model()
    if not model:
        return "Unidentified Insect"

    # --- Real inference would go here, e.g.: ---
    # img = cv2.resize(cropped_bgr_image, (224, 224))
    # img = img.astype("float32") / 255.0
    # img = np.expand_dims(img, axis=0)
    # preds = model.predict(img, verbose=0)
    # class_idx = np.argmax(preds[0])
    # return IP102_CLASS_NAMES[class_idx]

    return "Unidentified Insect"
