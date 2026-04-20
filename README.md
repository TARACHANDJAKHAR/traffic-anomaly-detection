# 🚦 Traffic Anomaly Detection — Local Demo

## Setup (takes ~2 minutes)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your trained models
After running the Kaggle notebook, download `traffic_anomaly_output.zip`.
Extract it and copy the `models/` folder here:

```
traffic_anomaly_demo/
  models/
    lstm_autoencoder.pt        ← required
    isolation_forest.pkl       ← required
    normalizer.npz             ← required
    scaler.pkl                 ← required
    combined_threshold.npy     ← required
    lstm_norm.npy              ← required
    if_norm.npy                ← required
  app.py
  inference.py
  model.py
  ...
```

### 3. Run the server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## Usage

- **Video tab** — drop any traffic MP4/AVI/MOV file
  - Get annotated video with bounding boxes + anomaly scores
  - See frame-by-frame anomaly timeline
  
- **Image tab** — drop any traffic photo
  - Get annotated image with per-vehicle scores
  - Instant results (no GPU needed)

---

## Notes

- YOLOv8n (~6 MB) downloads automatically on first run
- CPU-only inference — no GPU required for the demo
- Results saved in `static/results/` (auto-cleaned on restart if you want)
- Max upload: 500 MB video, 50 MB image