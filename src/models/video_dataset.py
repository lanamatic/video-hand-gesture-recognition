"""
PyTorch Dataset for the pre-extracted video clips (notebooks/clip_extraction.ipynb).

Clips on disk are (16, 112, 112, 3) uint8, range [0, 255]. This module turns
them to shape (C, T, H, W) = (3, 16, 112, 112), the PyTorch video convention.
Normalization: uint8 [0,255] -> float [0,1] -> per-channel mean/std (ImageNet
stats for the from-scratch model, Kinetics stats for the pretrained one).
Augmentation (train only): resize-then-crop jitter, horizontal flip,
temporal jitter, color jitter - all applied consistently across every frame.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import cv2

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

# Standard ImageNet stats - used for the from-scratch 3D CNN.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Kinetics-400 stats - what torchvision's R(2+1)D-18 was pretrained with
# (R2Plus1D_18_Weights.KINETICS400_V1.transforms()). Use these, not ImageNet,
# when feeding clips to that model.
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class GestureClipDataset(Dataset):
    """
    metadata_csv: path to _metadata.csv (has columns index, label, gesture_name, npy_path)
    train:        if True, apply random augmentation; if False, deterministic
    crop_size:    output spatial size after cropping (default 112)
    resize_to:    frames are resized to this before cropping (train jitter). 128 by default.
    """

    def __init__(self, metadata, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 train=False, resize_to=128, crop_size=112):
        if isinstance(metadata, (str, Path)):
            self.meta = pd.read_csv(metadata)
        else:
            self.meta = metadata.reset_index(drop=True)

        self.mean = mean.reshape(3, 1, 1, 1)
        self.std = std.reshape(3, 1, 1, 1)
        self.train = train
        self.resize_to = resize_to
        self.crop_size = crop_size

    def __len__(self):
        return len(self.meta)

    # Load the .npy clip
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        clip = np.load(row['npy_path'])  # (T, H, W, 3) uint8

        if self.train:
            clip = self._augment(clip)

        clip = clip.astype(np.float32) / 255.0 #[0,1]
        clip = np.transpose(clip, (3, 0, 1, 2))  # (3, T, H, W)
        clip = (clip - self.mean) / self.std

        return torch.from_numpy(clip).float(), int(row['label'])

    def _augment(self, clip):
        """
        Resize up then randomly crop back down - matches R(2+1)D-18's own
        pretraining recipe (resize 128, crop 112), which keeps the augmented
        distribution close to what the pretrained backbone already expects.
        Same crop offset/flip applied to every frame so motion stays coherent.
        """

        rs, cs = self.resize_to, self.crop_size

        if rs != clip.shape[1]:
            clip = np.stack([cv2.resize(f, (rs, rs)) for f in clip])

        # Random crop
        top = np.random.randint(0, rs - cs + 1)
        left = np.random.randint(0, rs - cs + 1)
        clip = clip[:, top:top + cs, left:left + cs, :]

        # Horizontal flip - same for every frame so the gesture stays coherent.
        if np.random.rand() < 0.5:
            clip = clip[:, :, ::-1, :]

        # Temporal jitter: shift the sampled window by a frame or two.
        if np.random.rand() < 0.3:
            shift = np.random.randint(-1, 2)
            clip = np.roll(clip, shift, axis=0)

        # Slight brightness/contrast jitter.
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.85, 1.15)
            clip = np.clip(clip.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(clip)


def stratified_train_val_split(metadata_csv, val_frac=0.15, seed=42):
    """
    Stratified validation set out of the train metadata only - never touches
    the official test split.
    """
    meta = pd.read_csv(metadata_csv)
    train_df, val_df = train_test_split(
        meta, test_size=val_frac, stratify=meta['label'], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_class_weights(metadata_df_or_csv, num_classes=13):
    """
    Balanced class weights for nn.CrossEntropyLoss(weight=...):
    weight_i = N / (num_classes * count_i), same formula as sklearn's
    class_weight='balanced'. 
    The expected weight over the training distribution is exactly 1,
    so the loss scale doesn't shift with imbalance.
    """
    if isinstance(metadata_df_or_csv, (str, Path)):
        meta = pd.read_csv(metadata_df_or_csv)
    else:
        meta = metadata_df_or_csv

    counts = meta['label'].value_counts().reindex(range(num_classes), fill_value=0)
    counts = counts.clip(lower=1)  # Avoid div by zero for an absent class
    weights = len(meta) / (num_classes * counts)
    return torch.tensor(weights.values, dtype=torch.float32)
