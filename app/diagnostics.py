import os
import sys
import numpy as np
import joblib
import torch

# Ensure we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import load_models
from app.inference import run_video_inference

def run_diagnostics(video_path=None):
    print("="*50)
    print("MODEL DIAGNOSTICS")
    print("="*50)
    
    # 1. Load artifacts directly to print their raw states
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    try:
        norm_data = np.load(os.path.join(model_dir, "normalizer.npz"))
        print("\n[Feature Normalizer]")
        print(f"  Means: {np.round(norm_data['mean'], 2)}")
        print(f"  Stds:  {np.round(norm_data['std'], 2)}")
        print(f"  Expected Center X: ~{norm_data['mean'][0]:.1f}")
        print(f"  Expected Center Y: ~{norm_data['mean'][1]:.1f}")
        print("  NOTE: Features must be scaled to 640x480 before normalization!")
    except Exception as e:
        print(f"Error loading normalizer: {e}")

    try:
        chk = torch.load(os.path.join(model_dir, "lstm_autoencoder.pt"), map_location="cpu", weights_only=False)
        print("\n[LSTM Autoencoder]")
        print(f"  Input Features: {chk.get('n_feat')}")
        print(f"  Sequence Length: {chk.get('seq_len')}")
        print(f"  Hidden Dim: {chk.get('hidden_dim')}")
        print(f"  Latent Dim: {chk.get('latent_dim')}")
    except Exception as e:
        print(f"Error loading LSTM checkpoint: {e}")

    try:
        l_norm = np.load(os.path.join(model_dir, "lstm_norm.npy"))
        print("\n[LSTM Score Normalization]")
        print(f"  Min Error: {l_norm[0]:.5f}")
        print(f"  Max Error: {l_norm[1]:.5f}")
    except Exception as e:
        print(f"Error loading LSTM norms: {e}")

    try:
        i_norm = np.load(os.path.join(model_dir, "if_norm.npy"))
        print("\n[Isolation Forest Normalization]")
        print(f"  Min Score: {i_norm[0]:.5f}")
        print(f"  Max Score: {i_norm[1]:.5f}")
    except Exception as e:
        print(f"Error loading IF norms: {e}")

    try:
        thresh = np.load(os.path.join(model_dir, "combined_threshold.npy"))
        print("\n[Combined Threshold]")
        print(f"  Anomaly Threshold: {thresh[0]:.5f}")
    except Exception as e:
        print(f"Error loading threshold: {e}")
        
    print("\n" + "="*50)
    
    if video_path:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
            
        print(f"VIDEO DIAGNOSTICS: {video_path}")
        print("="*50)
        
        try:
            bundle = load_models()
            temp_out = os.path.join(model_dir, "..", "static", "results", "diag_out.mp4")
            os.makedirs(os.path.dirname(temp_out), exist_ok=True)
            
            print("Processing video...")
            log = run_video_inference(video_path, temp_out, bundle, target_fps=5.0)
            
            total_frames = len(log)
            anomaly_frames = sum(1 for f in log if f["anomaly"])
            detections = sum(f.get("n_detections", 0) for f in log)
            
            print("\n[Results]")
            print(f"  Frames processed: {total_frames}")
            print(f"  Total detections: {detections}")
            print(f"  Anomaly frames:   {anomaly_frames} ({anomaly_frames/max(1, total_frames)*100:.1f}%)")
            
            if anomaly_frames / max(1, total_frames) > 0.5:
                print("  WARNING: High anomaly rate. Check for out-of-distribution issues or empty tracks.")
            else:
                print("  Status: Normal distribution observed.")
                
            if os.path.exists(temp_out):
                os.remove(temp_out)
                
        except Exception as e:
            print(f"Error processing video: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TrafficSense Model Diagnostics")
    parser.add_argument("--video", "-v", type=str, help="Path to test video", default=None)
    args = parser.parse_args()
    
    run_diagnostics(args.video)
