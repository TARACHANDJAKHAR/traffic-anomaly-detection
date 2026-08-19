"""
Traffic Anomaly Detection — Local Demo Server
"""

import os, json, time, uuid, shutil, threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(__name__, 
            template_folder=str(BASE_DIR / "app" / "templates"),
            static_folder=str(BASE_DIR / "app" / "static"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_SIZE", 500))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
RESULT_RETENTION_MINUTES = int(os.environ.get("RESULT_RETENTION_MINUTES", 30))

UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
RESULT_DIR = BASE_DIR / "app" / "static" / "results"
MODEL_DIR  = BASE_DIR / os.environ.get("MODEL_DIR", "models")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

jobs = {}

# ── Background Cleanup ────────────────────────────────────────────────────────
def cleanup_daemon():
    while True:
        now = time.time()
        max_age = RESULT_RETENTION_MINUTES * 60
        for d in [UPLOAD_DIR, RESULT_DIR]:
            if not d.exists(): continue
            for f in d.glob("*"):
                if f.is_file():
                    try:
                        if now - f.stat().st_mtime > max_age:
                            f.unlink()
                    except Exception as e:
                        print(f"Cleanup error for {f}: {e}")
        
        # Cleanup old jobs in memory
        expired_jobs = [jid for jid, j in jobs.items() if now - j.get("created_at", now) > max_age]
        for jid in expired_jobs:
            del jobs[jid]
            
        time.sleep(60 * 5)  # run every 5 mins

threading.Thread(target=cleanup_daemon, daemon=True).start()


# ── Lazy-load heavy deps ──────────────────────────────────────────────────────
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
            f"Missing model files in {MODEL_DIR}:\n" + "\n".join(missing)
        )

    import numpy as np
    import torch, joblib

    from app.model import LSTMAutoencoder, FeatureNormalizer

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
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

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


def background_video_inference(job_id, in_path, out_path, ext):
    temp_path = out_path.with_name(f"{job_id}_temp.mp4")
    try:
        bundle = load_models()
        from app.inference import run_video_inference
        
        def progress_cb(fidx, total):
            # Scale OpenCV processing to 0-80% progress
            jobs[job_id]["progress"] = round((fidx / total) * 80) if total else 0
            
        log = run_video_inference(str(in_path), str(temp_path), bundle, progress_callback=progress_cb)
        
        # FFmpeg Transcoding step (80-100%)
        jobs[job_id]["progress"] = 85
        import subprocess
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_path),
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_path)
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError("FFmpeg is not installed on the system.")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode('utf-8', errors='replace')
            raise RuntimeError(f"FFmpeg transcoding failed: {err}")
            
        n_frames  = len(log)
        n_anomaly = sum(1 for r in log if r["anomaly"])
        pct = round(n_anomaly / max(n_frames, 1) * 100, 1)
        
        jobs[job_id].update({
            "status": "done",
            "result_url": f"/static/results/{job_id}_output.mp4",
            "n_frames": n_frames,
            "n_anomaly": n_anomaly,
            "anomaly_pct": pct,
            "timeline": log,
            "progress": 100
        })
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        if in_path.exists(): 
            try: in_path.unlink()
            except: pass
        if temp_path.exists():
            try: temp_path.unlink()
            except: pass


@app.route("/infer/video", methods=["POST"])
def infer_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
        
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported extension {ext}. Allowed: {ALLOWED_VIDEO_EXTENSIONS}"}), 400

    job_id   = str(uuid.uuid4())[:8]
    in_path  = UPLOAD_DIR / f"{job_id}_input{ext}"
    out_path = RESULT_DIR / f"{job_id}_output.mp4"
    f.save(str(in_path))
    
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "created_at": time.time()
    }
    
    threading.Thread(target=background_video_inference, args=(job_id, in_path, out_path, ext), daemon=True).start()
    
    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/status/<job_id>")
def check_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/infer/image", methods=["POST"])
def infer_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
        
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported extension {ext}. Allowed: {ALLOWED_IMAGE_EXTENSIONS}"}), 400
        
    job_id   = str(uuid.uuid4())[:8]
    in_path  = UPLOAD_DIR / f"{job_id}_input{ext}"
    out_path = RESULT_DIR / f"{job_id}_output.jpg"
    f.save(str(in_path))

    try:
        bundle = load_models()
        from app.inference import run_image_inference
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
        if in_path.exists(): 
            try:
                in_path.unlink()
            except:
                pass


if __name__ == "__main__":
    print(f"\n🚦 Traffic Anomaly Detection — Local Demo")
    print("   Open: http://localhost:5000\n")
    if not any(MODEL_DIR.glob("*.pt")):
        print(f"⚠️  No models found in {MODEL_DIR}")
        print("   Copy your trained models/ folder here first.\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
