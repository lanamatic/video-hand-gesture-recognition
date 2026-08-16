"""
IPN Hand Dataset - loading and parsing anotations

Data structure:
    raw_deta/raw/
    ├── Annot_List.txt          # All annotations (with header)
    ├── Annot_TrainList.txt     # Train split (no header)
    ├── Annot_TestList.txt      # Test split (no header)
    ├── Video_TrainList.txt     # Train videos + total frame count
    ├── Video_TestList.txt      # Test videos + total frame count
    ├── classIdx.txt            # Mapping id → label code
    ├── metadata.csv            # Metadata
    ├── frames/                 # Frames (5 folders: frames, frames 2, ...)
        ├── 1CM1_1_R_#217/
        │   │   ├── frame000001.jpg
        │   │   ├── frame000002.jpg

Annotation format:
    video_name, label_code, class_id, start_frame, end_frame, n_frames

"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Constants

CLASS_ID_TO_NAME = {
    1:  "Non-gesture",
    2:  "Point-1f",
    3:  "Point-2f",
    4:  "Click-1f",
    5:  "Click-2f",
    6:  "Throw-up",
    7:  "Throw-down",
    8:  "Throw-left",
    9:  "Throw-right",
    10: "Open-twice",
    11: "DblClick-1f",
    12: "DblClick-2f",
    13: "Zoom-in",
    14: "Zoom-out"
}

# classIdx.txt
LABEL_CODE_TO_NAME = {
    "D0X": "Non-gesture",
    "B0A": "Point-1f",
    "B0B": "Point-2f",
    "G01": "Click-1f",
    "G02": "Click-2f",
    "G03": "Throw-up",
    "G04": "Throw-down",
    "G05": "Throw-left",
    "G06": "Throw-right",
    "G07": "Open-twice",
    "G08": "DblClick-1f",
    "G09": "DblClick-2f",
    "G10": "Zoom-in",
    "G11": "Zoom-out"
}

# Gesture only names - used for classification
GESTURE_NAMES = [
    "Point-1f", "Point-2f", "Click-1f", "Click-2f",
    "Throw-up", "Throw-down", "Throw-left", "Throw-right",
    "Open-twice", "DblClick-1f", "DblClick-2f", "Zoom-in", "Zoom-out"
]
NUM_CLASSES = 13 

# Anotation parsing

def parse_annotations(filepath, has_header = False):
    """
    Parse an IPN Hand annotation file
    Format: video_name,label_code,class_id,start_frame,end_frame,n_frames
    """
    records = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    start_idx = 1 if has_header else 0
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        if len(parts) < 6:
            continue
        
        video = parts[0]
        label_code = parts[1]
        class_id = int(parts[2])
        start_frame = int(parts[3])
        end_frame = int(parts[4])
        n_frames = int(parts[5])
        
        gesture_name = CLASS_ID_TO_NAME.get(class_id, f"Unknown-{class_id}")
        
        records.append({
            'video': video,
            'label_code': label_code,
            'class_id': class_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'n_frames': n_frames,
            'gesture_name': gesture_name,
        })
    
    return pd.DataFrame(records)

def load_train_test(data_dir="raw_data/raw"):
    """
    Load train and test annotations, filter out non-gesture class.
    """
    data_dir = Path(data_dir)
    
    train_df = parse_annotations(data_dir / "Annot_TrainList.txt", has_header=False)
    test_df = parse_annotations(data_dir / "Annot_TestList.txt", has_header=False)
    
    # Filter out non-gesture (class_id == 1)
    train_gestures = train_df[train_df['class_id'] != 1].copy()
    test_gestures = test_df[test_df['class_id'] != 1].copy()
    
    # Add 0-indexed label for ML models (class_id 2 → 0, class_id 3 → 1, ...)
    train_gestures['label'] = train_gestures['class_id'] - 2
    test_gestures['label'] = test_gestures['class_id'] - 2
    
    train_gestures = train_gestures.reset_index(drop=True)
    test_gestures = test_gestures.reset_index(drop=True)
    
    print(f"Train: {len(train_gestures)} gestures (from {len(train_df)} with non-gesture)")
    print(f"Test:  {len(test_gestures)} gestures (from {len(test_df)} with non-gesture)")
    print(f"Classes: {train_gestures['gesture_name'].nunique()} gestures")
    
    return train_gestures, test_gestures

def load_all_annotations(data_dir="raw_data/raw"):
    """Load all annotations (Annot_List.txt with header)."""
    return parse_annotations(Path(data_dir) / "Annot_List.txt", has_header=True)

# Working with frames

def find_frames_dirs(data_dir="raw_data/raw"):
    """
    Find all frame directories.
    IPN Hand comes in 5 folders: frames, frames 2, frames 3, frames 4, frames 5
    Returns list of Paths to directories containing video subfolders
    """
    data_dir = Path(data_dir)
    
    frame_dirs = []
    for name in ["frames", "frames 2", "frames 3", "frames 4", "frames 5"]:
        d = data_dir / name
        if d.exists() and d.is_dir():
            frame_dirs.append(d)
    
    if not frame_dirs:
        # Fallback: find any directory with video subfolders
        for d in data_dir.iterdir():
            if d.is_dir() and any(d.iterdir()):
                subdirs = [x for x in d.iterdir() if x.is_dir()]
                if subdirs:
                    frame_dirs.append(d)
    
    print(f"Found {len(frame_dirs)} frame directories: {[d.name for d in frame_dirs]}")
    return frame_dirs

def find_video_dir(video_name, frame_dirs):
    """
    Search for a video's folder across all frame directories.
    Returns path to the video folder, or None
    """
    for fd in frame_dirs:
        video_dir = fd / video_name
        if video_dir.exists():
            return video_dir
    return None

def get_frame_paths(video_dir, start_frame, end_frame):
    """
    Return list of frame file paths for one gesture.
    """
    video_name = video_dir.name  # e.g. "1CM1_3_R_#226"
    
    paths = []
    for i in range(start_frame, end_frame + 1):
        # IPN Hand format: {video_name}_{NNNNNN}.jpg
        p = video_dir / f"{video_name}_{i:06d}.jpg"
        if p.exists():
            paths.append(p)
    return paths

def load_gesture_frames(video_name, start_frame, end_frame, frame_dirs,
                        max_frames=None):
    """
    Load frames for one gesture as a numpy array.
    If max_frames is set, uniformly sample down to this many frames
    """
    import cv2
    
    video_dir = find_video_dir(video_name, frame_dirs)
    if video_dir is None:
        return None
    
    paths = get_frame_paths(video_dir, start_frame, end_frame)
    if not paths:
        return None
    
    # Uniform sampling if needed
    if max_frames and len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        paths = [paths[i] for i in indices]
    
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    
    return np.array(frames) if frames else None

# Statistics

def print_stats(df, name="Dataset"):
    """Print dataset statistics."""
    
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total samples:   {len(df)}")
    print(f"  Unique videos:   {df['video'].nunique()}")
    print(f"  Num classes:     {df['gesture_name'].nunique()}")
    
    print(f"\n  Class distribution:")
    counts = df['gesture_name'].value_counts()
    max_count = counts.max()
    for cls_name, count in counts.items():
        bar_len = int(count / max_count * 30)
        print(f"    {cls_name:14s} {count:5d} ")
    
    print(f"\n  Gesture duration (frames, ~30fps):")
    print(f"    Min:    {df['n_frames'].min():6d}  ({df['n_frames'].min()/30:.1f}s)")
    print(f"    Max:    {df['n_frames'].max():6d}  ({df['n_frames'].max()/30:.1f}s)")
    print(f"    Mean:   {df['n_frames'].mean():6.1f}  ({df['n_frames'].mean()/30:.1f}s)")
    print(f"    Median: {df['n_frames'].median():6.1f}  ({df['n_frames'].median()/30:.1f}s)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_data/raw"
    
    train_df, test_df = load_train_test(data_dir)
    print_stats(train_df, "TRAIN (gestures only)")
    print_stats(test_df, "TEST (gestures only)")
    
    frame_dirs = find_frames_dirs(data_dir)
    if frame_dirs:
        sample = train_df.iloc[0]
        vdir = find_video_dir(sample['video'], frame_dirs)
        print(f"Sample video: {sample['video']}")
        print(f"  Directory: {vdir}")
        if vdir:
            paths = get_frame_paths(vdir, sample['start_frame'], sample['end_frame'])
            print(f"  Frames found: {len(paths)} / {sample['n_frames']}")
