import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config.firebase import download_model_if_needed

MAX_FRAMES = 20
FRAME_HEIGHT = 160
FRAME_WIDTH = 160
CLASS_NAMES = ['J', 'Z']

MODEL_PATH = "models/3dcnn_efficient.h5"
FIREBASE_MODEL_PATH = "models/3dcnn_efficient.h5"  # Path di Firebase Storage

def load_dynamic_model():
    download_model_if_needed(MODEL_PATH, FIREBASE_MODEL_PATH)
    return load_model(MODEL_PATH)

def predict_video(file_path, model):
    video_data, error = preprocess_video(file_path)
    if error:
        return {
            "label": error,
            "confidence": 0.0,
            "raw_label": None
        }

    preds = model.predict(video_data, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = float(preds[class_id])
    raw_label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"

    print(f"[INFO] Video Prediction: {raw_label} ({confidence*100:.1f}%)")

    return {
        "label": f"{raw_label} ({confidence*100:.1f}%)",
        "confidence": confidence,
        "raw_label": raw_label
    }
