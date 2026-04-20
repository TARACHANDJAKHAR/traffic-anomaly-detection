"""
Traffic Anomaly Detection — Local Demo Server
Run: python app.py
Then open: http://localhost:5000
"""

import os, json, time, uuid, shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_DIR = Path("static/uploads")
RESULT_DIR = Path("static/results")
MODEL_DIR  = Path("models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Lazy-load heavy deps only when inference is called ────────────────────────
_models_loaded = False
_model_bundle  = {}

def load_models():
    global _models_loaded, _model_bundle
    if _models_loaded:
        return _model_bundle

    required = ["lstm_autoencoder.pt", "isolation_forest.pkl",
                "normalizer.npz", "scaler.pkl",
                "combined_threshold.npy", "lstm_norm.npy", "if_norm.npy"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Copy your trained models/ folder from Kaggle into this directory."
        )

    import numpy as np
    import torch, joblib

    # Import model classes (inline so Flask starts even without torch)
    from model import LSTMAutoencoder, FeatureNormalizer

    chk = torch.load(str(MODEL_DIR/"lstm_autoencoder.pt"),
                     map_location="cpu", weights_only=False)
    lstm = LSTMAutoencoder(
        chk["n_feat"], chk["hidden_dim"], chk["latent_dim"],
        chk["num_layers"], chk["dropout"], chk["seq_len"]
    )
    lstm.load_state_dict(chk["model_state"]); lstm.eval()

    norm = FeatureNormalizer()
    norm.load(str(MODEL_DIR/"normalizer.npz"))

    _model_bundle = {
        "lstm":      lstm,
        "norm":      norm,
        "iso":       joblib.load(str(MODEL_DIR/"isolation_forest.pkl")),
        "scaler":    joblib.load(str(MODEL_DIR/"scaler.pkl")),
        "l_min":     float(np.load(str(MODEL_DIR/"lstm_norm.npy"))[0]),
        "l_max":     float(np.load(str(MODEL_DIR/"lstm_norm.npy"))[1]),
        "i_min":     float(np.load(str(MODEL_DIR/"if_norm.npy"))[0]),
        "i_max":     float(np.load(str(MODEL_DIR/"if_norm.npy"))[1]),
        "threshold": float(np.load(str(MODEL_DIR/"combined_threshold.npy"))[0]),
        "seq_len":   chk["seq_len"],
        "n_feat":    chk["n_feat"],
    }
    _models_loaded = True
    print("✓ Models loaded")
    return _model_bundle


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models/status")
def model_status():
    required = ["lstm_autoencoder.pt", "isolation_forest.pkl",
                "normalizer.npz", "scaler.pkl",
                "combined_threshold.npy", "lstm_norm.npy", "if_norm.npy"]
    present  = [f for f in required if (MODEL_DIR / f).exists()]
    missing  = [f for f in required if not (MODEL_DIR / f).exists()]
    return jsonify({
        "ready":   len(missing) == 0,
        "present": present,
        "missing": missing,
        "loaded":  _models_loaded,
    })

@app.route("/infer/video", methods=["POST"])
def infer_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id   = str(uuid.uuid4())[:8]
    ext      = Path(f.filename).suffix.lower()
    in_path  = UPLOAD_DIR / f"{job_id}_input{ext}"
    out_path = RESULT_DIR / f"{job_id}_output.mp4"
    f.save(str(in_path))

    try:
        bundle = load_models()
        from inference import run_video_inference
        log = run_video_inference(str(in_path), str(out_path), bundle)
        n_frames  = len(log)
        n_anomaly = sum(1 for r in log if r["anomaly"])
        pct = round(n_anomaly / max(n_frames, 1) * 100, 1)
        return jsonify({
            "job_id":      job_id,
            "result_url":  f"/static/results/{job_id}_output.mp4",
            "n_frames":    n_frames,
            "n_anomaly":   n_anomaly,
            "anomaly_pct": pct,
            "timeline":    log,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if in_path.exists(): in_path.unlink()

@app.route("/infer/image", methods=["POST"])
def infer_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    job_id   = str(uuid.uuid4())[:8]
    ext      = Path(f.filename).suffix.lower() or ".jpg"
    in_path  = UPLOAD_DIR / f"{job_id}_input{ext}"
    out_path = RESULT_DIR / f"{job_id}_output.jpg"
    f.save(str(in_path))

    try:
        bundle = load_models()
        from inference import run_image_inference
        result = run_image_inference(str(in_path), str(out_path), bundle)
        return jsonify({
            "job_id":      job_id,
            "result_url":  f"/static/results/{job_id}_output.jpg",
            "detections":  result["detections"],
            "n_anomalies": result["n_anomalies"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if in_path.exists(): in_path.unlink()

@app.route("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory("static/results", filename)

if __name__ == "__main__":
    print("\n🚦 Traffic Anomaly Detection — Local Demo")
    print("   Open: http://localhost:5000\n")
    if not any(MODEL_DIR.glob("*.pt")):
        print("⚠️  No models found in ./models/")
        print("   Copy your trained models/ folder from Kaggle here first.\n")
    app.run(debug=True, port=5000)
