"""
MediaPipe hand landmark extraction.

Extracts 21 hand landmarks (x, y, z) per frame using MediaPipe Hands.
Each frame → 63 features (21 landmarks x 3 coordinates).
A gesture (sequence of frames) → array of shape (T, 63).
"""

import os
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import sys
from src.data.dataset import load_train_test, find_frames_dirs

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3

def create_hands_detector(static_mode=False, max_hands=1, min_confidence=0.5):
    
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("mediapipe not installed. Run: pip install mediapipe")
    
    return mp.solutions.hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=max_hands,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    )

def extract_landmarks_from_frame(frame_rgb, hands_detector):
  
    results = hands_detector.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)

def extract_landmarks_from_gesture(video_name, start_frame, end_frame,
                                   frame_dirs, hands_detector):

    from src.data.dataset import find_video_dir, get_frame_paths

    video_dir = find_video_dir(video_name, frame_dirs)
    if video_dir is None:
        return None, 0.0

    frame_paths = get_frame_paths(video_dir, start_frame, end_frame)
    if not frame_paths:
        return None, 0.0

    sequence = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landmarks = extract_landmarks_from_frame(img_rgb, hands_detector)
        if landmarks is not None:
            sequence.append(landmarks)

    if not sequence:
        return None, 0.0

    detection_rate = len(sequence) / len(frame_paths)
    return np.array(sequence, dtype=np.float32), detection_rate

# Normalize landmark sequence
def normalize_landmarks(sequence):

    seq = sequence.copy().reshape(-1, NUM_LANDMARKS, 3)  # (T, 21, 3)

    xy = seq[:, :, :2]   # (T, 21, 2)
    z = seq[:, :, 2:3]   # (T, 21, 1)

    # Translation: subtract wrist (landmark 0)
    wrist_xy = xy[:, 0:1, :]  # (T, 1, 2)
    xy = xy - wrist_xy

    # Scale: distance from wrist (now origin) to middle-finger MCP (landmark 9)
    mcp_xy = xy[:, 9, :]  # (T, 2)
    scale = np.linalg.norm(mcp_xy, axis=1, keepdims=True)  # (T, 1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    xy = xy / scale[:, np.newaxis, :]

    seq = np.concatenate([xy, z], axis=2)  # (T, 21, 3)
    return seq.reshape(-1, NUM_FEATURES)  # (T, 63)

# Extract landmarks for entire dataset split and save as .npy files
def extract_dataset_landmarks(df, frame_dirs, output_dir, normalize=True):
    
    os.makedirs(output_dir, exist_ok=True)

    detector = create_hands_detector(static_mode=False, max_hands=1)

    records = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting landmarks"):
        seq, det_rate = extract_landmarks_from_gesture(
            row['video'], row['start_frame'], row['end_frame'],
            frame_dirs, detector
        )

        if seq is None or len(seq) == 0:
            skipped += 1
            continue

        if normalize:
            seq = normalize_landmarks(seq)

        npy_path = os.path.join(output_dir, f"{idx:05d}_class{row['label']}.npy")
        np.save(npy_path, seq)

        records.append({
            'index': idx,
            'label': row['label'],
            'gesture_name': row['gesture_name'],
            'n_frames_valid': len(seq),
            'detection_rate': det_rate,
            'npy_path': npy_path,
        })

    detector.close()

    meta = pd.DataFrame(records)
    meta_path = os.path.join(output_dir, "_metadata.csv")
    meta.to_csv(meta_path, index=False)

    print(f"\nDone. Extracted {len(records)} gestures, skipped {skipped} "
          f"(no hand detected).")
    print(f"Mean detection rate: {meta['detection_rate'].mean():.1%}")
    print(f"Metadata saved to: {meta_path}")

    return meta