# Signify ML API

Signify ML API adalah layanan inference machine learning untuk menerjemahkan gestur Bahasa Isyarat Indonesia (SIBI) menjadi teks secara real-time. Proyek ini merupakan bagian dari capstone project Coding Camp 2025 dan berfungsi sebagai backend modular untuk menangani klasifikasi citra dan video gestur menggunakan dua model terpisah.

## 🚀 Fitur Utama

- 🔤 Klasifikasi huruf statis SIBI (A-I, K-Y) dengan EfficientNetB0
- 🔁 Klasifikasi huruf dinamis (J & Z) dengan 3D CNN
- 📸 Dukungan input gambar & video (upload atau real-time)
- ⚡ Inference cepat via REST API & WebSocket
- 🔧 Modular: layanan model dipisahkan agar efisien
- 🧪 Model dilatih dan dievaluasi untuk kebutuhan real-time

## 🧠 Arsitektur Model

### 1. Model EfficientNetB0 (Statis)
- Dataset: Gambar gestur tangan SIBI A-I, K-Y
- Preprocessing: MediaPipe Hand Detection, resizing, normalization
- Framework: TensorFlow / Keras
- Format model: `.h5`

### 2. Model 3D CNN (Dinamis)
- Dataset: Video gestur SIBI J dan Z
- Preprocessing: Ekstraksi frame, resizing, normalisasi
- Arsitektur: 3D Convolutional Neural Network
- Format model: `.h5`

## 🛠 Teknologi

- Python 3.10+
- FastAPI
- TensorFlow / Keras
- MediaPipe
- OpenCV
- Uvicorn (server)
- WebSocket

## 🔌 Endpoint Utama

| Method | Endpoint                 | Deskripsi                             |
|--------|--------------------------|----------------------------------------|
| POST   | `/predict/image`         | Prediksi huruf dari gambar             |
| POST   | `/predict/video`         | Prediksi huruf dari video              |
| WS     | `/ws/predict-realtime`   | Inference real-time dengan webcam      |
| GET    | `/`                      | Health check                           |

## 📦 Struktur Project
```bash
signify-ml-api/
├── config/
│ ├── firebase.py (configurasi Firebase)
├── labels/
│ ├── label_map_v2.json (labels)
├── models/
│ └── efficientnet_signify_v2.h5 (Untuk memprediksi image statis)
│ └── efficientnet_signify.h5 (Untuk memprediksi realtime)
│ └── 3dcnn_efficient.h5 (Untuk memprediksi video)
├── utils/
│ └── hand_detector.py
│ └── image_inference.py
│ └── realtime_inference.py
│ └── video_inference.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Menjalankan Secara Lokal

1. **Clone Repo**
    ```bash
    git clone https://github.com/yudhriz/signify-ml-api.git
    cd signify-ml-api
    ```
2. **Buat environment**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Jalankan server**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

## 🌍 Deployment
Layanan ini di-deploy di Hugging Face Space dan dikonfigurasi untuk mendukung inference gambar dan video melalui HTTP dan WebSocket.

## 📄 Lisensi
Proyek ini bersifat open-source dan menggunakan lisensi MIT. Silakan gunakan dan kembangkan untuk tujuan edukasi maupun sosial.

### 📬 Jika ada pertanyaan, silakan hubungi kami melalui LinkedIn atau buat issue di repo ini.
