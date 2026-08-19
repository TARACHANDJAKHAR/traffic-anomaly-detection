import numpy as np
import joblib

print("--- NORMALIZER ---")
d = np.load("models/normalizer.npz")
mean = d["mean"]
std = d["std"]
features = ["cx", "cy", "w", "h", "speed", "accel", "heading", "heading_change", "iou"]
for i, f in enumerate(features):
    print(f"{f}: mean={mean[i]:.2f}, std={std[i]:.2f}")

print("\n--- LSTM THRESHOLDS ---")
lstm_norm = np.load("models/lstm_norm.npy")
print(f"LSTM Min: {lstm_norm[0]:.5f}, Max: {lstm_norm[1]:.5f}")

print("\n--- IF THRESHOLDS ---")
if_norm = np.load("models/if_norm.npy")
print(f"IF Min: {if_norm[0]:.5f}, Max: {if_norm[1]:.5f}")

print("\n--- COMBINED THRESHOLD ---")
thresh = np.load("models/combined_threshold.npy")
print(f"Combined Threshold: {thresh[0]:.5f}")
