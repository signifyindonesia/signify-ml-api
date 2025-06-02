import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector
from config.firebase import download_model_if_needed

MODEL_PATH = "models/efficientnet_signify_v2.h5"
FIREBASE_MODEL_PATH = "models/efficientnet_signify_v2.h5"
LABELS_PATH = "labels/label_map.json"
CONFIDENCE_THRESHOLD = 0.5

def pad_to_square(image):
    h, w, _ = image.shape
    size = max(h, w)
    delta_w = size - w
    delta_h = size - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def load_static_model():

    download_model_if_needed(MODEL_PATH, FIREBASE_MODEL_PATH)

    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    detector = HandDetector()
    return model, labels, detector

def predict_static_image(image, model, labels):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0]
        class_id = int(np.argmax(pred))
        confidence = float(pred[class_id])
        raw_label = labels[class_id] if class_id < len(labels) else "Unknown"

        return {
            "label": f"{raw_label} ({confidence*100:.1f}%)",
            "confidence": confidence,
            "raw_label": raw_label
        }

    except Exception as e:
        return {
            "label": f"Prediction error: {str(e)}",
            "confidence": 0.0,
            "raw_label": None
        }


