"""
Landmark sequence dataset for the RNN models.
Loads the (T, 63) sequences produced by landmark_extractor.py, splits train/val
by subject, and turns them into fixed-length tensors the RNN can batch.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 13
NUM_FEATURES = 63

# Load one split -> (sequences, labels, metadata).
# Drops gestures that are too short or poorly detected.
def load_landmark_split(landmarks_dir, min_frames=4, min_detection_rate=0.0):
    meta = pd.read_csv(f"{landmarks_dir}/_metadata.csv")
    meta = meta[meta['n_frames_valid'] >= min_frames]
    meta = meta[meta['detection_rate'] >= min_detection_rate].reset_index(drop=True)

    sequences, keep = [], []
    for i, row in meta.iterrows():
        try:
            seq = np.load(row['npy_path'])
        except FileNotFoundError:
            fname = str(row['npy_path']).split('/')[-1]
            seq = np.load(f"{landmarks_dir}/{fname}")
        sequences.append(seq.astype(np.float32))
        keep.append(i)

    meta = meta.loc[keep].reset_index(drop=True)
    labels = meta['label'].to_numpy()
    return sequences, labels, meta

# Add source video + subject id to the metadata, matched on the original index.
def attach_video_names(meta, df):
    meta = meta.copy()
    meta['video'] = meta['index'].map(df['video'])
    meta['subject'] = meta['video'].str.split('_').str[0]
    return meta

# Split train/val BY SUBJECT -> returns (train_mask, val_mask).
# The same person in both sets would make the model recognise the person, not the gesture.
def subject_wise_split(meta, val_fraction=0.2, seed=42):
    subjects = np.array(sorted(meta['subject'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:n_val])

    val_mask = meta['subject'].isin(val_subjects).to_numpy()
    return ~val_mask, val_mask

# Append frame-to-frame differences: (T, 63) -> (T, 126).
# Position encodes hand shape, velocity encodes motion (Throw-up vs Throw-down).
def add_velocity(seq):
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=1).astype(np.float32)

# Per-feature mean/std over all frames of the training split.
def fit_scaler(sequences):
    stacked = np.concatenate(sequences, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

# Landmark sequences -> fixed-length tensors, keeping the true length
# so pack_padded_sequence can ignore the padding.
class GestureSequenceDataset(Dataset):
    def __init__(self, sequences, labels, t_fixed, mean=None, std=None,
                 use_velocity=True):
        self.t_fixed = t_fixed
        self.mean, self.std = mean, std
        self.use_velocity = use_velocity

        self.sequences, self.labels, self.lengths = [], [], []
        for seq, label in zip(sequences, labels):
            if self.use_velocity:
                seq = add_velocity(seq)
            if self.mean is not None:
                seq = (seq - self.mean) / self.std

            T = len(seq)
            if T > t_fixed:                                  # uniform sampling
                idx = np.linspace(0, T - 1, t_fixed).astype(int)
                seq, length = seq[idx], t_fixed
            else:                                            # zero padding
                pad = np.zeros((t_fixed - T, seq.shape[1]), dtype=np.float32)
                seq, length = np.concatenate([seq, pad], axis=0), T

            self.sequences.append(torch.from_numpy(seq.astype(np.float32)))
            self.labels.append(int(label))
            self.lengths.append(length)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# Inverse-frequency weights for CrossEntropyLoss.
# Point-1f / Point-2f have ~5x more samples; without this the model just predicts them.
def compute_class_weights(labels, num_classes=NUM_CLASSES):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

# Build the three loaders in one call.
# Scaler is fitted on TRAIN ONLY - fitting on everything would leak test statistics.
def build_loaders(train_seqs, train_labels, val_seqs, val_labels,
                  test_seqs, test_labels, t_fixed, batch_size=64,
                  use_velocity=True, standardize=True):
    prep = [add_velocity(s) for s in train_seqs] if use_velocity else train_seqs
    mean, std = fit_scaler(prep) if standardize else (None, None)

    def make(seqs, labels, shuffle):
        ds = GestureSequenceDataset(seqs, labels, t_fixed, mean, std, use_velocity)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (make(train_seqs, train_labels, True),
            make(val_seqs, val_labels, False),
            make(test_seqs, test_labels, False))
