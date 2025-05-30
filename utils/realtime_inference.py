# utils/realtime_inference.py
import cv2
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector
from config.firebase import download_model_if_needed

MODEL_PATH = "models/efficientnet_signify.h5"
FIREBASE_MODEL_PATH = "models/efficientnet_signify.h5"  # Path di Firebase Storage
LABELS_PATH = "labels/label_map.json"

def load_realtime_resources():
    # Cek dan unduh model dari Firebase jika belum tersedia
    download_model_if_needed(MODEL_PATH, FIREBASE_MODEL_PATH)

    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    detector = HandDetector()
    return model, labels, detector

def predict_frame(frame, model, labels, detector):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_drawing, bbox = detector.find_hand(frame)

    if bbox:
        x1, y1, x2, y2 = bbox
        hand_roi = rgb_image[y1:y2, x1:x2]

        if hand_roi.size > 0:
            try:
                img = cv2.resize(hand_roi, (224, 224))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)[0]
                class_id = np.argmax(pred)
                confidence = float(pred[class_id])
                label = labels[class_id] if class_id < len(labels) else "Unknown"

                print(f"[INFO] Predicted: {label} ({confidence:.2f})")
                return label, confidence

            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
                return "Prediction Error", 0.0

    return "No hand", 0.0

