"""
MediaPipe hand landmark extraction.

Extracts 21 hand landmarks (x, y, z) per frame using MediaPipe Hands.
Each frame → 63 features (21 landmarks x 3 coordinates).
A gesture (sequence of frames) → array of shape (T, 63).
"""

import os
import numpy as np 
import pandas as pd 

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