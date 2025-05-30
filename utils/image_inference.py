import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector
from config.firebase import download_model_if_needed

MODEL_PATH = "models/efficientnet_signify.h5"
FIREBASE_MODEL_PATH = "models/efficientnet_signify.h5"  # Path di Firebase Storage
LABELS_PATH = "labels/label_map.json"

def load_static_model():
    # Unduh model dari Firebase jika belum tersedia secara lokal
    download_model_if_needed(MODEL_PATH, FIREBASE_MODEL_PATH)

    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    detector = HandDetector()
    return model, labels, detector

def predict_static_image(image, model, labels, detector):
    if image is None:
        raise ValueError("Gambar tidak valid atau gagal dibaca (image=None)")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_with_drawing, bbox = detector.find_hand(rgb_image)

    if bbox:
        x1, y1, x2, y2 = bbox
        hand_roi = rgb_image[y1:y2, x1:x2]

        if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
            try:
                img = cv2.resize(hand_roi, (224, 224))
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

    return {
        "label": "No hand detected",
        "confidence": 0.0,
        "raw_label": None
    }
