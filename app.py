# File: app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import numpy as np
import cv2

from utils.image_inference import load_static_model, predict_static_image
from utils.video_inference import load_dynamic_model, predict_video
from utils.realtime_inference import load_realtime_resources, predict_frame
from config.firebase import upload_image_to_firebase

app = FastAPI(title="Signify ML Service")

# === CORS (ubah allow_origins jika frontend punya domain khusus) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load models once at startup ===
image_model, image_labels, image_detector = load_static_model()
realtime_model, realtime_labels, realtime_detector = load_realtime_resources()
video_model = load_dynamic_model()

# === Endpoints ===
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File harus berupa gambar (jpg/png)")

    content = await file.read()
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        # ✅ Baca image sebagai numpy array sebelum diprediksi
        image = cv2.imread(tmp_path)
        if image is None:
            raise ValueError("Gagal membaca gambar. Format mungkin tidak valid atau file rusak.")

        prediction = predict_static_image(image, image_model, image_labels)
        image_url = upload_image_to_firebase(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi atau upload: {str(e)}")
    finally:
        os.remove(tmp_path)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "prediction": prediction,
            "image_url": image_url
        }
    )

@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/avi"]:
        raise HTTPException(status_code=400, detail="File harus berupa video (mp4/avi)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        prediction = predict_video(tmp_path, video_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video prediction error: {str(e)}")
    finally:
        os.remove(tmp_path)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "prediction": prediction
        }
    )

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Realtime WebSocket connected.")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                await websocket.send_json({"error": "Empty frame data"})
                continue

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            label, confidence = predict_frame(frame, realtime_model, realtime_labels, realtime_detector)

            await websocket.send_json({
                "prediction": label,
                "confidence": confidence
            })
    except WebSocketDisconnect:
        print("[INFO] Client disconnected from realtime websocket")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
