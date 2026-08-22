"""
Full landmark extraction for both splits.
Runs from the project root:
    python3 src/data/run_extraction.py
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from src.data.dataset import load_train_test, find_frames_dirs
from src.data.landmark_extractor import (
    create_hands_detector, extract_landmarks_from_gesture, normalize_landmarks,
)

DATA_DIR = "raw_data/raw"
OUT_DIR = "data/landmarks"
FLUSH_EVERY = 25


def extract_split(df, frame_dirs, output_dir, normalize=True):
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "_metadata.csv")

    # Resume: keep whatever was already extracted
    if os.path.exists(meta_path):
        records = pd.read_csv(meta_path).to_dict('records')
        done = {r['index'] for r in records}
        print(f"  resuming - {len(done)} gestures already extracted")
    else:
        records, done = [], set()

    todo = df[~df.index.isin(done)]
    if todo.empty:
        print("  nothing left to do")
        return pd.DataFrame(records)

    detector = create_hands_detector(static_mode=False, max_hands=1)
    skipped = 0

    for n, (idx, row) in enumerate(tqdm(todo.iterrows(), total=len(todo),
                                        desc=f"  {os.path.basename(output_dir)}"), 1):
        seq, det_rate = extract_landmarks_from_gesture(
            row['video'], row['start_frame'], row['end_frame'], frame_dirs, detector)

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

        if n % FLUSH_EVERY == 0:
            pd.DataFrame(records).to_csv(meta_path, index=False)

    detector.close()

    meta = pd.DataFrame(records)
    meta.to_csv(meta_path, index=False)

    print(f"  extracted {len(meta)}, skipped {skipped} (no hand detected)")
    if len(meta):
        print(f"  mean detection rate: {meta['detection_rate'].mean():.1%}")
    return meta


def main():
    train_df, test_df = load_train_test(DATA_DIR)
    frame_dirs = find_frames_dirs(DATA_DIR)

    print("\nTRAIN")
    train_meta = extract_split(train_df, frame_dirs, f"{OUT_DIR}/train")
    print("\nTEST")
    test_meta = extract_split(test_df, frame_dirs, f"{OUT_DIR}/test")

    print(f"\nDone. train={len(train_meta)}  test={len(test_meta)}")


if __name__ == "__main__":
    main()