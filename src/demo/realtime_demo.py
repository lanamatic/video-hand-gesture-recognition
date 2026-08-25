"""
Real-time hand gesture recognition demo (webcam).

Uses the trained BiLSTM model over MediaPipe landmark sequences.

Pipeline per frame:
    webcam frame -> MediaPipe (21 landmarks) -> normalize -> add to rolling buffer
    when buffer has T_fixed frames -> add velocity -> standardize -> BiLSTM -> prediction
    show gesture on screen if confidence is high enough (with temporal smoothing)

IMPORTANT — feature pipeline must match training exactly:
    1. normalize_landmarks   (same as landmark_extractor.py)
    2. add_velocity          (63 -> 126, np.diff with prepend)
    3. standardize           (subtract train mean, divide by train std)
The scaler (mean/std) MUST be the one fitted on the training set (hosted on Google Drive).

Run:  python src/demo/realtime_demo.py
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '.')   # so `from src...` works when run from repo root

import time
from collections import deque, Counter

import numpy as np
import cv2
import torch

from src.data.landmark_extractor import (
    create_hands_detector, extract_landmarks_from_frame, normalize_landmarks,
    NUM_FEATURES
)
from src.models.lstm_model import GestureRNN   # adjust import to teammate's filename
from src.models.landmark_dataset import add_velocity

# Config (must match the trained model)
CKPT_PATH = "data/checkpoints/best_BiLSTM.pth"
SCALER_PATH = "data/checkpoints/scaler.npz"

T_FIXED = 60           # config: T_fixed
HIDDEN = 128           # config: hidden
LAYERS = 2             # config: layers
DROPOUT = 0.3          # config: dropout
BIDIRECTIONAL = True   # BiLSTM
USE_VELOCITY = True    # config: velocity_features = true -> 126 input features
INPUT_SIZE = NUM_FEATURES * 2 if USE_VELOCITY else NUM_FEATURES   # 126

CONF_THRESHOLD = 0.6   # Only show a gesture above this probability
SMOOTH_WINDOW = 5      # Majority vote over the last N predictions
PREDICT_EVERY = 3      # Run the model every N frames (not every frame)

GESTURE_NAMES = [
    "Point-1f", "Point-2f", "Click-1f", "Click-2f",
    "Throw-up", "Throw-down", "Throw-left", "Throw-right",
    "Open-twice", "DblClick-1f", "DblClick-2f", "Zoom-in", "Zoom-out"
]

def load_model(device):
    model = GestureRNN(
        input_size=INPUT_SIZE, hidden_size=HIDDEN, num_layers=LAYERS,
        num_classes=len(GESTURE_NAMES), rnn_type='lstm',
        bidirectional=BIDIRECTIONAL, dropout=DROPOUT,
    ).to(device)

    sd = torch.load(CKPT_PATH, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model + scaler
    model = load_model(device)
    scaler = np.load(SCALER_PATH)
    mean, std = scaler['mean'], scaler['std']
    print(f"Loaded model and scaler (mean/std shape {mean.shape})")

    # MediaPipe detector (video mode = uses tracking, faster)
    detector = create_hands_detector(static_mode=False, max_hands=1)

    # Rolling buffer of normalized landmark frames (each is 63-dim)
    buffer = deque(maxlen=T_FIXED)
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

        frame = cv2.flip(frame, 1)   # mirror, feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract landmarks for this frame
        landmarks = extract_landmarks_from_frame(rgb, detector)
        hand_present = landmarks is not None

        if hand_present:
            # normalize a single (1, 63) frame, take row 0 -> (63,)
            norm = normalize_landmarks(landmarks.reshape(1, -1))[0]
            buffer.append(norm)
        else:
            buffer.clear()          # reset when hand leaves the frame
            recent_preds.clear()
            current_label = ""

        frame_count += 1

        # Predict every PREDICT_EVERY frames, once the buffer is full
        if hand_present and len(buffer) == T_FIXED and frame_count % PREDICT_EVERY == 0:
            seq = np.array(buffer, dtype=np.float32)          # (60, 63)
            if USE_VELOCITY:
                seq = add_velocity(seq)                        # (60, 126)
            seq = (seq - mean) / std                           # standardize

            x = torch.from_numpy(seq).float().unsqueeze(0).to(device)  # (1, 60, 126)
            lengths = torch.tensor([T_FIXED])

            with torch.no_grad():
                logits = model(x, lengths)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred = int(probs.argmax())
            conf = float(probs[pred])

            if conf >= CONF_THRESHOLD:
                recent_preds.append(pred)

            # Temporal smoothing: majority vote
            if recent_preds:
                smoothed = Counter(recent_preds).most_common(1)[0][0]
                current_label = GESTURE_NAMES[smoothed]
                current_conf = conf
            else:
                current_label = ""

        # ---- Draw overlay ----
        h, w = frame.shape[:2]
        # buffer fill bar
        fill = len(buffer) / T_FIXED
        cv2.rectangle(frame, (10, h - 30), (10 + int(300 * fill), h - 15),
                      (0, 200, 0), -1)
        cv2.rectangle(frame, (10, h - 30), (310, h - 15), (255, 255, 255), 1)

        if current_label:
            text = f"{current_label}  ({current_conf:.0%})"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
        elif not hand_present:
            cv2.putText(frame, "No hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (200, 200, 0), 2)

        cv2.imshow("Hand Gesture Recognition (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()