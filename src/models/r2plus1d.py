"""
R(2+1)D-18 pretrained on Kinetics-400, fine-tuned for the 13 gesture classes.

Actually trained on Kaggle (see notebooks/07_2_kaggle_modelR(2+1)D.ipynb) - a 31M
parameter video model is too slow to fine-tune on local MPS. This module holds
the same architecture so the checkpoint trained on Kaggle can be loaded back
in here for evaluation once it's downloaded.
"""

import torch.nn as nn
import torchvision.models.video as video_models


def build_r2plus1d(num_classes=13, dropout=0.5):
    try:
        weights = video_models.R2Plus1D_18_Weights.KINETICS400_V1
        model = video_models.r2plus1d_18(weights=weights)
    except AttributeError:
        model = video_models.r2plus1d_18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model
