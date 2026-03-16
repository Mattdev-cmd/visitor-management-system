# Visitor Management & Emotion Recognition System

> **AI-powered visitor management for schools / facilities, running on a Raspberry Pi 5 (2 GB).**

| Feature | Description |
|---|---|
| **Visitor Registration** | Name, purpose, contact, photo capture |
| **Facial Recognition** | Identifies returning visitors automatically |
| **Emotion Recognition** | Detects Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust |
| **Live Camera Scanner** | Real-time face detection & analysis from the dashboard |
| **Digital Database** | SQLite-backed visitor records, images, and emotion logs |
| **Monitoring Dashboard** | Stats, visitor logs, emotion charts, search |
| **Auto-Logging** | Timestamped, automated entry + emotion recording |

---

## Architecture

```
┌──────────────┐      HTTP / MJPEG       ┌────────────────────────┐
│   Browser    │ ◄──────────────────────► │   Flask Web Server     │
│  Dashboard   │                          │   (app.py)             │
└──────────────┘                          │                        │
                                          │  ┌──────────────────┐  │
                                          │  │ Face Recognition  │  │
  ┌──────────┐   capture frame            │  │ (dlib / HOG)     │  │
  │ PiCamera │ ──────────────────────►    │  └──────────────────┘  │
  │ / USB    │                            │  ┌──────────────────┐  │
  └──────────┘                            │  │ Emotion CNN      │  │
                                          │  │ (TFLite, 48×48)  │  │
                                          │  └──────────────────┘  │
                                          │  ┌──────────────────┐  │
                                          │  │ SQLite Database   │  │
                                          │  │ (visitors.db)    │  │
                                          │  └──────────────────┘  │
                                          └────────────────────────┘
```

---

## Project Structure

```
visitor-management-system/
├── app.py                      # Flask entry-point & routes
├── config.py                   # All configuration in one place
├── database.py                 # SQLite schema & helpers
├── camera.py                   # PiCamera2 / OpenCV camera abstraction
├── requirements.txt
├── README.md
│
├── services/
│   ├── face_service.py         # Face detection, encoding, matching
│   ├── emotion_service.py      # Emotion CNN inference (TFLite / Keras)
│   └── visitor_service.py      # Visitor CRUD & visit logging
│
├── models/
│   ├── train_emotion_model.py  # Training script (FER-2013)
│   ├── emotion_model.tflite    # (generated after training)
│   └── emotion_model.h5        # (generated after training)
│
├── static/
│   ├── css/style.css
│   ├── js/dashboard.js
│   └── uploads/                # Visitor photos
│
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── register.html
    ├── visitors.html
    └── visitor_detail.html
```

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Board | Raspberry Pi 5 (2 GB RAM) |
| Camera | Pi Camera Module 3 *or* USB webcam |
| Storage | 16 GB+ microSD (Class 10 / A2) |
| OS | Raspberry Pi OS (64-bit, Bookworm) |
| Power | 27 W USB-C supply |
| Network | Wi-Fi or Ethernet (for dashboard access) |

---

## Software Setup

### 1. Flash & configure the Pi

```bash
# After first boot:
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv cmake libatlas-base-dev \
    libhdf5-dev libjpeg-dev libopenblas-dev
```

### 2. Clone / copy the project

```bash
cd ~
git clone <your-repo-url> visitor-management-system
cd visitor-management-system
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `dlib` and `face-recognition` compile from source on ARM64 —
> this takes a while but only needs to happen once. Make sure `cmake` is
> installed.

### 5. (Optional) Train the emotion model

Download `fer2013.csv` from
[Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
and place it in the project root, then:

```bash
python models/train_emotion_model.py --csv fer2013.csv --epochs 35
```

This produces `models/emotion_model.tflite`. Without it, the system
falls back to a heuristic estimator so the application still runs.

### 6. Run the application

```bash
python app.py
```

Open a browser and go to **`http://<pi-ip>:5000`**.

---

## Configuration

All settings live in `config.py` and can be overridden with environment
variables:

| Env Variable | Default | Purpose |
|---|---|---|
| `VMS_SECRET_KEY` | random | Flask secret key |
| `VMS_DEBUG` | `false` | Enable Flask debug mode |
| `VMS_HOST` | `0.0.0.0` | Bind address |
| `VMS_PORT` | `5000` | HTTP port |
| `VMS_CAMERA_BACKEND` | `opencv` | `opencv` or `picamera` |
| `VMS_CAMERA_INDEX` | `0` | Camera index for OpenCV |

---

## Usage Guide

### Registering a Visitor

1. Click **Register Visitor** in the sidebar.
2. Fill in name, purpose, and contact.
3. Start the camera and capture a photo (or upload one).
4. Submit — the system enrols the face and logs the visit with an
   emotion reading.

### Dashboard Monitoring

- **Stats cards** show total visitors, visits today, currently-inside
  count, and today's top emotion.
- **Live Scanner** lets you scan any face in real time — it identifies
  known visitors and detects emotions.
- **Quick Check-In** button appears for recognised visitors.
- **Recent Activity** table shows the latest visits with emotions and
  checkout controls.

### Searching Visitors

Go to **Visitor Logs** and use the search bar to filter by name.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/scan` | Send `{ "image": "<base64>" }` → face + emotion results |
| `POST` | `/api/quick_checkin` | `{ "visitor_id", "emotion", "confidence" }` → auto check-in |
| `GET`  | `/api/stats` | Dashboard statistics JSON |

---

## Performance Notes (2 GB Pi 5)

- **HOG face detector** is used by default (`config.FACE_RECOGNITION_MODEL = "hog"`)
  — fast on CPU, ~200 ms per frame.
- **TFLite emotion model** with quantisation keeps RAM under 50 MB.
- **SQLite WAL mode** enables concurrent reads from the dashboard while
  the camera writes.
- The browser-side camera (`getUserMedia`) offloads video rendering from
  the Pi — only captured JPEG frames are sent to the server for AI
  analysis.

---

## Security Considerations

- Face encodings are stored as opaque blobs (pickle); they cannot reconstruct the original face.
- The Flask `SECRET_KEY` should be set via the `VMS_SECRET_KEY` environment variable in production.
- File uploads are size-limited (5 MB) and restricted to image extensions.
- SQL queries use parameterised statements — no injection risk.
- For production deployments, place the app behind a reverse proxy
  (nginx) with HTTPS.

---

## License

MIT — free for academic and personal use.
