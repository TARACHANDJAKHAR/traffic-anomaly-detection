import numpy as np
import cv2
import torch
from app.model import LSTMAutoencoder, FeatureNormalizer
from app.inference import pad_to_seq, score_sequence
import joblib

norm = FeatureNormalizer()
norm.load("models/normalizer.npz")
chk = torch.load("models/lstm_autoencoder.pt", map_location="cpu", weights_only=False)
lstm = LSTMAutoencoder(chk["n_feat"], chk["hidden_dim"], chk["latent_dim"],
                       chk["num_layers"], chk["dropout"], chk["seq_len"])
lstm.load_state_dict(chk["model_state"])
lstm.eval()

l_min, l_max = np.load("models/lstm_norm.npy")
iso = joblib.load("models/isolation_forest.pkl")
sc = joblib.load("models/scaler.pkl")
i_min, i_max = np.load("models/if_norm.npy")
thresh = float(np.load("models/combined_threshold.npy")[0])

print(f"Threshold: {thresh:.4f}")

bundle = {
    "norm": norm, "lstm": lstm, "l_min": l_min, "l_max": l_max,
    "iso": iso, "scaler": sc, "i_min": i_min, "i_max": i_max,
    "threshold": thresh, "seq_len": chk["seq_len"]
}

# Let's create a perfectly normal real-world trajectory:
# Car driving straight down the road at 30mph.
# Over 20 frames at 5 FPS (4 seconds total), it moves a lot.
hist = []
# Assuming 640x480 frame. Car starts at y=400 (bottom), moves to y=200 (middle) in 20 frames.
# That's dy = -10 per frame.
cx = 320.0
cy = 400.0
w, h = 100.0, 80.0
fps = 5.0

# Initialize track state logic manually
prev_cx, prev_cy = cx, cy
prev_speed = 0.0
prev_heading = 0.0

for i in range(20):
    cy -= 10.0 # move up
    w *= 0.95 # getting smaller as it moves away
    h *= 0.95
    
    if i == 0:
        speed, accel, heading, hc = 0.0, 0.0, 0.0, 0.0
    else:
        dx, dy = cx - prev_cx, cy - prev_cy
        speed = float(np.sqrt(dx**2 + dy**2) * fps)
        accel = (speed - prev_speed) * fps
        heading = float(np.degrees(np.arctan2(dy, dx)))
        hc = heading - prev_heading
        if hc > 180: hc -= 360
        if hc < -180: hc += 360
    
    prev_cx, prev_cy = cx, cy
    prev_speed, prev_heading = speed, heading
    
    iou = 0.0
    feat = np.array([cx, cy, w, h, speed, accel, heading, hc, iou], dtype=np.float32)
    hist.append(feat)
    
seq = pad_to_seq(hist, 20)
score = score_sequence(seq, bundle)
print(f"Simulated normal car score: {score:.4f}")
print("Sequence features (frame 10):", seq[10])

# Simulated synthetic normal
hist_synth = []
rng = np.random.default_rng(42)
cx, cy = 320, 240
vx, vy = rng.uniform(-3, 3), rng.uniform(-1, 1)
w, h = 75, 45
ps = float(np.hypot(vx, vy)*5)
ph = float(np.degrees(np.arctan2(vy, vx)))
for t in range(20):
    vx += rng.normal(0, 0.1)
    vy += rng.normal(0, 0.05)
    cx = np.clip(cx+vx, 30, 610)
    cy = np.clip(cy+vy, 30, 450)
    sp = float(np.hypot(vx, vy)*5)
    ac = sp - ps
    hd = float(np.degrees(np.arctan2(vy, vx)))
    hc = hd - ph
    if hc > 180: hc -= 360
    if hc < -180: hc += 360
    feat = np.array([cx, cy, w, h, sp, ac, hd, hc, rng.uniform(0, 0.15)], dtype=np.float32)
    hist_synth.append(feat)
    ps, ph = sp, hd

seq_synth = pad_to_seq(hist_synth, 20)
score_synth = score_sequence(seq_synth, bundle)
print(f"Simulated synthetic normal score: {score_synth:.4f}")
