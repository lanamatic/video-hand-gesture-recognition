"""
Real-time hand gesture recognition demo (webcam) — R(2+1)D-18 version.

Uses the trained R(2+1)D-18 model over raw video clips (pixels, no landmarks).

Pipeline:
    webcam frames -> rolling buffer of 16 frames -> resize 112x112 -> normalize
    -> (1, 3, 16, 112, 112) -> R(2+1)D-18 -> prediction
    show gesture if confidence is high enough (with temporal smoothing)

NOTE: R(2+1)D-18 is heavy. On Mac MPS each inference takes a few hundred ms,
so prediction is not per-frame — it runs every PREDICT_EVERY frames. Expect
visible latency compared to the landmark (BiLSTM) demo. This is expected.


Run:  python src/demo/realtime_demo_r2plus1d.py
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '.')

import time
from collections import deque, Counter

import numpy as np
import cv2
import torch

from src.models.r2plus1d import build_r2plus1d


# ---------------- Config (must match training) ----------------
CKPT_PATH = "data/checkpoints/best_r2plus1d.pth"   # adjust to your path

N_FRAMES = 16          # clip length used in training
SIZE = 112             # spatial resolution

CONF_THRESHOLD = 0.6   # only show a gesture above this probability
SMOOTH_WINDOW = 5      # majority vote over last N predictions
PREDICT_EVERY = 8      # run the (heavy) model every N frames

# ImageNet stats used when clips were prepared (must match training exactly)
IMAGENET_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
IMAGENET_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

GESTURE_NAMES = [
    "Point-1f", "Point-2f", "Click-1f", "Click-2f",
    "Throw-up", "Throw-down", "Throw-left", "Throw-right",
    "Open-twice", "DblClick-1f", "DblClick-2f", "Zoom-in", "Zoom-out"
]


def load_model(device):
    model = build_r2plus1d(num_classes=len(GESTURE_NAMES)).to(device)
    sd = torch.load(CKPT_PATH, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()
    return model


def preprocess_clip(frames_list):
    """
    frames_list: list of 16 RGB frames (H, W, 3) uint8.
    Returns tensor (1, 3, 16, 112, 112) ready for the model.
    """
    # resize each frame to 112x112
    clip = np.stack([cv2.resize(f, (SIZE, SIZE)) for f in frames_list])  # (16,112,112,3)
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    clip = np.transpose(clip, (3, 0, 1, 2))   # (3, 16, 112, 112)
    return torch.from_numpy(clip).float().unsqueeze(0)   # (1, 3, 16, 112, 112)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    print("Loaded R(2+1)D-18 model.")

    # Rolling buffer of raw RGB frames
    buffer = deque(maxlen=N_FRAMES)
    recent_preds = deque(maxlen=SMOOTH_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frame_count = 0
    current_label = ""
    current_conf = 0.0

    print("Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb)
        frame_count += 1

        # Predict every PREDICT_EVERY frames, once buffer is full
        if len(buffer) == N_FRAMES and frame_count % PREDICT_EVERY == 0:
            x = preprocess_clip(list(buffer)).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred = int(probs.argmax())
            conf = float(probs[pred])

            if conf >= CONF_THRESHOLD:
                recent_preds.append(pred)

            if recent_preds:
                smoothed = Counter(recent_preds).most_common(1)[0][0]
                current_label = GESTURE_NAMES[smoothed]
                current_conf = conf
            else:
                current_label = ""

        # ---- Draw overlay ----
        h, w = frame.shape[:2]
        fill = len(buffer) / N_FRAMES
        cv2.rectangle(frame, (10, h - 30), (10 + int(300 * fill), h - 15),
                      (0, 200, 0), -1)
        cv2.rectangle(frame, (10, h - 30), (310, h - 15), (255, 255, 255), 1)

        if current_label:
            text = f"{current_label}  ({current_conf:.0%})"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (200, 200, 0), 2)

        cv2.imshow("Hand Gesture Recognition - R(2+1)D-18 (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()