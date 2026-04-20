"""
Inference functions for video and image input.
Called by Flask routes in app.py.
"""
import numpy as np
import torch
import cv2
from collections import defaultdict, deque
from dataclasses import dataclass

VEHICLE_CLASSES  = [0, 1, 2, 3, 5, 7]
MIN_FRAMES_FOR_LSTM = 8


# ── Track state ───────────────────────────────────────────────────────────────
@dataclass
class TrackState:
    prev_cx:      float = 0.0
    prev_cy:      float = 0.0
    prev_speed:   float = 0.0
    prev_heading: float = 0.0
    initialized:  bool  = False


def compute_iou(box, others):
    if len(others) == 0:
        return 0.0
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area  = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return float(np.max(inter / (area + areas - inter + 1e-6)))


def update_track(state, box, others, fps):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h   = x2 - x1, y2 - y1
    if not state.initialized:
        state.prev_cx = cx; state.prev_cy = cy; state.initialized = True
        speed = accel = heading = heading_change = 0.0
    else:
        dx, dy = cx - state.prev_cx, cy - state.prev_cy
        speed  = float(np.sqrt(dx**2 + dy**2) * fps)
        accel  = (speed - state.prev_speed) * fps
        heading = float(np.degrees(np.arctan2(dy, dx)))
        heading_change = heading - state.prev_heading
        if heading_change >  180: heading_change -= 360
        if heading_change < -180: heading_change += 360
        state.prev_cx = cx; state.prev_cy = cy
        state.prev_speed = speed; state.prev_heading = heading
    iou = compute_iou(box, others)
    return np.array([cx, cy, w, h, speed, accel, heading, heading_change, iou],
                    dtype=np.float32)


def flatten_sequences(X):
    delta = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), delta], axis=1)


def pad_to_seq(history, seq_len):
    if len(history) < MIN_FRAMES_FOR_LSTM:
        return None
    arr = np.stack(history, axis=0)
    if len(arr) < seq_len:
        pad = np.zeros((seq_len - len(arr), arr.shape[1]), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr[-seq_len:]


def make_tracker():
    import supervision as sv
    try:
        return sv.ByteTrack(minimum_matching_threshold=0.8, minimum_consecutive_frames=1)
    except TypeError:
        try:
            return sv.ByteTrack(track_activation_threshold=0.30, lost_track_buffer=20,
                                minimum_matching_threshold=0.8)
        except TypeError:
            return sv.ByteTrack()


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_sequence(seq_arr, bundle):
    norm  = bundle["norm"]; lstm = bundle["lstm"]
    l_min = bundle["l_min"]; l_max = bundle["l_max"]
    iso   = bundle["iso"];   sc   = bundle["scaler"]
    i_min = bundle["i_min"]; i_max = bundle["i_max"]

    seq_n = norm.transform(seq_arr[np.newaxis])
    t_in  = torch.from_numpy(seq_n).float()
    err   = lstm.reconstruction_error(t_in).item()
    ls    = float(np.clip((err - l_min) / (l_max - l_min + 1e-8), 0, 1))

    flat    = flatten_sequences(seq_n)
    flat_sc = sc.transform(flat)
    ir      = -iso.score_samples(flat_sc)[0]
    ifs     = float(np.clip((ir - i_min) / (i_max - i_min + 1e-8), 0, 1))

    return 0.6 * ls + 0.4 * ifs


def score_to_color(score):
    if score < 0.4:    return (0, 200, 0)    # green
    elif score < 0.65: return (0, 140, 255)  # orange
    else:              return (0, 0, 220)    # red


def draw_box(frame, box, label, score, is_anomaly):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = score_to_color(score)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_anomaly else 1)
    txt = f"{'[!] ' if is_anomaly else ''}{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ── Video inference ───────────────────────────────────────────────────────────
def run_video_inference(video_path: str, output_path: str, bundle: dict,
                         target_fps: float = 5.0) -> list:
    from ultralytics import YOLO
    import supervision as sv

    yolo    = YOLO("yolov8n.pt")
    tracker = make_tracker()
    thresh  = bundle["threshold"]
    seq_len = bundle["seq_len"]

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip  = max(1, int(round(native_fps / target_fps)))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             native_fps, (W, H))

    t_states = defaultdict(TrackState)
    t_hists  = defaultdict(lambda: deque(maxlen=seq_len + 5))
    t_scores = {}
    frame_log = []
    fidx = n_anom = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fidx += 1
        if fidx % skip != 0:
            writer.write(frame); continue

        res  = yolo(frame, imgsz=416, conf=0.30, classes=VEHICLE_CLASSES,
                    verbose=False, device="cpu")[0]
        dets = sv.Detections.from_ultralytics(res)
        frame_anomaly = False

        if len(dets) > 0:
            tracks = tracker.update_with_detections(dets)
            if len(tracks) > 0 and tracks.tracker_id is not None:
                aboxes = tracks.xyxy
                for i, tid in enumerate(tracks.tracker_id):
                    box    = tracks.xyxy[i]
                    others = np.delete(aboxes, i, 0)
                    t_hists[tid].append(
                        update_track(t_states[tid], box, others, target_fps))
                    score = t_scores.get(tid, 0.0)
                    hist  = list(t_hists[tid])
                    if len(hist) >= MIN_FRAMES_FOR_LSTM:
                        seq_arr = pad_to_seq(hist, seq_len)
                        if seq_arr is not None:
                            score = score_sequence(seq_arr, bundle)
                            t_scores[tid] = score
                    is_an = score >= thresh
                    if is_an: n_anom += 1; frame_anomaly = True
                    draw_box(frame, box, f"ID:{tid}", score, is_an)

        # HUD
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (240, 50), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Frame {fidx}/{total}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Anomalies: {n_anom}", (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 220), 1)
        writer.write(frame)
        frame_log.append({"frame": fidx, "anomaly": frame_anomaly,
                           "n_detections": len(dets)})

    cap.release(); writer.release()
    return frame_log


# ── Image inference ───────────────────────────────────────────────────────────
def run_image_inference(image_path: str, output_path: str, bundle: dict) -> dict:
    from ultralytics import YOLO
    import supervision as sv
    from sklearn.preprocessing import StandardScaler

    frame  = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    iso    = bundle["iso"]; sc = bundle["scaler"]
    i_min  = bundle["i_min"]; i_max = bundle["i_max"]
    thresh = bundle["threshold"] * 0.85   # lower bar for single-frame mode
    H_f, W_f = frame.shape[:2]

    yolo = YOLO("yolov8n.pt")
    res  = yolo(frame, imgsz=640, conf=0.30, classes=VEHICLE_CLASSES,
                verbose=False, device="cpu")[0]
    dets = sv.Detections.from_ultralytics(res)
    ann  = frame.copy()
    detections_out = []

    if len(dets) == 0:
        cv2.putText(ann, "No vehicles detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        n_anom = 0
    else:
        boxes = dets.xyxy
        feats = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h   = x2 - x1, y2 - y1
            iou    = compute_iou(boxes[i], np.delete(boxes, i, 0))
            f_raw  = np.array([cx, cy, w, h, 0, 0, 0, 0, iou], dtype=np.float32)
            feats.append(np.concatenate([f_raw] * 5))  # replicate to match IF input dim
        feats = np.stack(feats)
        try:
            feats_sc = sc.transform(feats)
        except Exception:
            _tmp = StandardScaler().fit(feats)
            feats_sc = _tmp.transform(feats)

        raw   = -iso.score_samples(feats_sc)
        scores = np.clip((raw - i_min) / (i_max - i_min + 1e-8), 0, 1)
        n_anom = 0
        for i, (box, score) in enumerate(zip(boxes, scores)):
            is_an = float(score) >= thresh
            if is_an: n_anom += 1
            draw_box(ann, box, f"V{i+1}", float(score), is_an)
            detections_out.append({
                "box":        [int(v) for v in box],
                "score":      round(float(score), 3),
                "is_anomaly": bool(is_an),
            })

        # Legend
        for yi, (col, lbl) in enumerate([
            ((0, 200, 0),   "Normal"),
            ((0, 140, 255), "Borderline"),
            ((0, 0, 220),   "Anomaly"),
        ]):
            yy = ann.shape[0] - 90 + yi * 28
            cv2.rectangle(ann, (10, yy), (36, yy + 22), col, -1)
            cv2.putText(ann, lbl, (44, yy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imwrite(output_path, ann)
    return {"detections": detections_out, "n_anomalies": n_anom}
