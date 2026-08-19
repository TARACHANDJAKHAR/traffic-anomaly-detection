<div align="center">
  <h1>🚦 TrafficSense — Traffic Anomaly Detection</h1>
  <p>
    <b>An experimental computer vision pipeline for identifying anomalous traffic behavior.</b>
  </p>
  
  [![Live Demo](https://img.shields.io/badge/Live_Demo-traffic--anomaly--detection.onrender.com-success?style=for-the-badge&logo=render)](https://traffic-anomaly-detection.onrender.com)
</div>

TrafficSense is a Flask-based computer vision research and demo system designed to analyze traffic scenes and identify potentially anomalous vehicle behavior. By decoupling spatial object detection from temporal trajectory analysis, the system identifies unusual movements—such as sudden braking, wrong-way driving, or erratic swerving—using a combination of deep learning and statistical methods.

<div align="center">
  <img src="app/static/uploads/input.jpg" width="48%" alt="Traffic Input Frame" />
  <img src="app/static/results/18fb834b_output.jpg" width="48%" alt="Traffic Anomaly Detection Output" />
  <p><i>Left: Input image. Right: YOLO detection identifying the vehicle (Score: 0.63), correctly scoring it as normal.</i></p>
</div>

> [!WARNING]
> **Honesty Disclaimer**: This is an experimental traffic anomaly detection pipeline, not a production-grade or safety-critical enforcement system. Performance heavily depends on the training data, scene characteristics, and camera angles. Anomaly scores are model-dependent, and YOLO detection alone does not mean a vehicle is anomalous.

---

## 🔬 Machine Learning Pipeline

The core ML architecture operates in distinct phases to separate "seeing" from "understanding."

1. **YOLOv8 Detection**: The pipeline uses YOLOv8 to detect vehicles (cars, trucks, buses, etc.) in the frame. *Note: YOLO detection alone does not mean that a vehicle is anomalous.*
2. **Tracking (ByteTrack)**: ByteTrack assigns persistent IDs to detections, allowing the system to follow individual vehicles across multiple frames.
3. **Temporal Features**: Video anomaly detection requires temporal information. The pipeline extracts trajectory-related features from the tracks (e.g., relative velocity, acceleration, heading changes, and bounding box scale).
4. **LSTM Autoencoder**: An LSTM network encodes and reconstructs the sequential trajectory. High reconstruction errors signal behaviors the model has rarely seen (e.g., sudden U-turns).
5. **Isolation Forest**: A secondary statistical model that evaluates flattened trajectory sequences against normal traffic flow parameters.
6. **Score Fusion**: The final per-vehicle anomaly score is computed as a weighted fusion: `0.6 × LSTM + 0.4 × Isolation Forest`.

---

## 🖼️ Image vs. 🎥 Video Analysis

TrafficSense supports two distinct inference modes:

### 🖼️ Image Analysis
Because a single image lacks temporal sequence data, image mode focuses entirely on object detection and per-object spatial analysis. It evaluates bounding box characteristics rather than claiming temporal behavior.

### 🎥 Video Analysis
Video analysis evaluates the full temporal pipeline:
`Upload video` → `background processing` → `tracking` → `temporal analysis` → `anomaly scoring` → `annotated output` → `frame timeline`

*(Note: Long-running video inference utilizes asynchronous processing and job polling to prevent blocking the web server.)*

---

## 🌐 Web Application

The system exposes a modern web interface and several API endpoints powered by Flask:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Returns a basic 200 OK status to verify the server is running. |
| `/models/status` | `GET` | Checks if all required `.pt`, `.pkl`, and `.npy` model artifacts are present and loaded. |
| `/infer/image` | `POST` | Accepts an image upload, runs spatial detection, and returns an annotated image URL. |
| `/infer/video` | `POST` | Accepts a video upload, initiates background processing, and returns a `job_id`. |
| `/status/<job_id>` | `GET` | Polls the current processing status and progress of a background video job. |

---

## 📁 Project Structure

```text
traffic-anomaly-detection/
├── app/
│   ├── app.py
│   ├── diagnostics.py
│   ├── inference.py
│   ├── model.py
│   ├── static/
│   └── templates/
├── models/
├── Main_notebook.ipynb
├── inspect_model.py
├── nb_extracted.py
├── test_app.py
├── test_docker.py
├── test_scores.py
├── Dockerfile
├── requirements.txt
├── yolov8n.pt
└── README.md
```

---

## 🚀 Getting Started

### Clone the Repository
```bash
git clone https://github.com/TARACHANDJAKHAR/traffic-anomaly-detection.git
cd traffic-anomaly-detection
```

### Install Dependencies
Create a virtual environment, then install the required Python packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Locally
Launch the Flask application from the project root:
```bash
python3 -m app.app
```
Navigate to `http://localhost:5000` in your browser.

---

## 🐳 Docker

The application is fully containerized for easy deployment.

**Build the image:**
```bash
docker build -t trafficsense .
```

**Run the container:**
```bash
docker run --rm -p 5000:5000 -e PORT=5000 trafficsense
```
*(The Dockerfile uses the `$PORT` environment variable to bind Gunicorn).*

---

## 📦 Model Artifacts

The `models/` directory contains the trained weights and parameters required by the pipeline. *These are experimental research artifacts.*

- `lstm_autoencoder.pt`: PyTorch weights for the temporal reconstruction model.
- `isolation_forest.pkl`: Scikit-Learn weights for the statistical novelty detector.
- `normalizer.npz`: Feature normalization parameters.
- `scaler.pkl`: Standard scaler parameters for the Isolation Forest.
- `combined_threshold.npy`, `lstm_threshold.npy`: Float thresholds used to flag anomalies.
- `lstm_norm.npy`, `if_norm.npy`: Min/max values used to normalize the raw scores to a 0.0–1.0 scale.

---

## 🧪 Testing

The repository includes scripts to validate the pipeline:
- `test_app.py`: Tests the Flask application endpoints and file upload mechanisms.
- `test_docker.py`: Validates the container build and dependencies.
- `test_scores.py`: A diagnostic script used to test the LSTM and Isolation Forest feature boundaries.

*(These tests have been used for validation, though complete test coverage is not claimed).*

---

## ⚠️ Limitations

- **Training Dependency**: Anomaly detection quality heavily depends on the training data distribution.
- **Scene Variance**: Unusual camera angles, extremely close viewpoints, or non-standard scenes can affect detection accuracy.
- **Temporal History**: Temporary anomaly detection requires a sufficient video history (minimum sequence lengths) to establish baseline velocity and acceleration.
- **Hardware Bottlenecks**: CPU inference can be slower for long or high-resolution videos; the pipeline currently processes at a targeted FPS to mitigate this.
- **Scope**: This is currently a research/demo system rather than a safety-critical traffic enforcement system.

---

## 🔮 Future Improvements

- Training the LSTM on a broader dataset to improve generalization.
- Adding GPU acceleration support for the background video processing threads.
- Implementing richer event-level explanations (e.g., differentiating between "Stopped" vs. "Wrong Way").
- Adding persistent object storage (like AWS S3) for uploaded and processed videos in a cloud deployment.
- Quantitative benchmark evaluations against standardized datasets.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core backend language |
| **PyTorch** | Deep learning framework (LSTM Autoencoder) |
| **YOLOv8** | Spatial object detection |
| **OpenCV** | Video processing, frame extraction, and annotation |
| **Flask** | Web server and API routing |
| **scikit-learn** | Statistical modeling (Isolation Forest) |
| **Docker** | Containerization and deployment |
| **HTML/CSS/JS** | Frontend web interface |

---

## 👨‍💻 Author

**Tarachand Jakhar**  
B.Tech — Artificial Intelligence  
SVNIT Surat  
[GitHub Profile](https://github.com/TARACHANDJAKHAR)