import os
import uuid
from tqdm import tqdm
import firebase_admin
from firebase_admin import credentials, storage

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
    size = blob.size

    with open(model_path, "wb") as f:
        with tqdm(total=size, unit='B', unit_scale=True, desc=os.path.basename(model_path)) as pbar:
            def progress(current, total):
                pbar.update(current - pbar.n)
            blob.download_to_file(f, raw_download=True, timeout=300, start=0, end=None, progress_callback=progress)

    print(f"[INFO] Selesai mengunduh: {model_path}")

def ensure_models_downloaded():
    """Pastikan semua model sudah terunduh saat startup."""
    download_model_if_needed("model/efficientnet_signify.h5", "models/efficientnet_signify.h5")
    download_model_if_needed("model/3dcnn_efficient.h5", "models/3dcnn_efficient.h5")


def upload_image_to_firebase(file_path, folder="uploaded_images"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} tidak ditemukan.")

    # Nama file unik
    filename = f"{folder}/{uuid.uuid4().hex}_{os.path.basename(file_path)}"

    # Upload ke Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_path)

    # Buat URL publik
    blob.make_public()
    return blob.public_url
