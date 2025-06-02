import os
import uuid
from tqdm import tqdm
import firebase_admin
from firebase_admin import credentials, storage
import mimetypes

FIREBASE_STORAGE_BUCKET = 'signify-indonesia.firebasestorage.app'
SERVICE_ACCOUNT_PATH = "serviceAccount.json"

# Inisialisasi Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': FIREBASE_STORAGE_BUCKET
    })

bucket = storage.bucket()

def download_model_if_needed(model_path: str, firebase_path: str):
    """Unduh model dari Firebase jika belum tersedia secara lokal, dengan progress bar."""
    if os.path.exists(model_path):
        print(f"[INFO] Model '{model_path}' sudah tersedia.")
        return

    print(f"[INFO] Mengunduh model '{model_path}' dari Firebase...")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    blob = bucket.blob(firebase_path)
    if not blob.exists():
        raise FileNotFoundError(f"Model '{firebase_path}' tidak ditemukan di Firebase Storage.")

    size = blob.size or 0

    with open(model_path, "wb") as f:
        blob.download_to_file(f)

    print(f"[INFO] Selesai mengunduh: {model_path}")

    return model_path

def ensure_models_downloaded():
    print("[INFO] Memastikan semua model telah tersedia...")
    download_model_if_needed("models/efficientnet_signify_v2.h5", "models/efficientnet_signify_v2.h5")
    download_model_if_needed("models/efficientnet_signify.h5", "models/efficientnet_signify.h5")
    download_model_if_needed("models/3dcnn_efficient.h5", "models/3dcnn_efficient.h5")

def upload_image_to_firebase(file_path, folder="uploaded_images"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} tidak ditemukan.")
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image"):
        raise ValueError("Hanya file gambar yang boleh diupload.")

    # Nama file unik
    filename = f"{folder}/{uuid.uuid4().hex}_{os.path.basename(file_path)}"

    # Upload ke Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_path, content_type="image/jpeg")

    # Buat URL publik
    blob.make_public()
    return blob.public_url
