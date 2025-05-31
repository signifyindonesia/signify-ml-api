import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config.firebase import download_model_if_needed
import os

MAX_FRAMES = 20
FRAME_HEIGHT = 160
FRAME_WIDTH = 160
CLASS_NAMES = ['J', 'Z']

MODEL_PATH = "models/3dcnn_efficient.h5"
FIREBASE_MODEL_PATH = "models/3dcnn_efficient.h5"  # Path di Firebase Storage

def load_dynamic_model():
    download_model_if_needed(MODEL_PATH, FIREBASE_MODEL_PATH)
    return load_model(MODEL_PATH)

def preprocess_video(file_path):
    if not os.path.exists(file_path):
        return None, "File not found"

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None, "No frames found in video"

    frame_indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) < MAX_FRAMES:
        return None, "Not enough valid frames in video"

    video_array = np.stack(frames, axis=0)  # (20, 160, 160, 3)
    video_array = np.expand_dims(video_array, axis=0)  # (1, 20, 160, 160, 3)
    return video_array, None

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
