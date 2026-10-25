"""
Exact PyTorch and Vision Model Architectures for Rock & Mineral Classification.
Preserved exactly from the original Montanuniversität Leoben research notebooks.
"""

import torch
import torch.nn as nn
from torchvision import models

# Fixed class order used during training across all 1fps and 30fps experiments
CLASS_NAMES = sorted(['Albit', 'Calcite', 'Dolomit', 'Feldspat', 'Quarz', 'Rhodocrosite', 'Tile'])
NUM_CLASSES = len(CLASS_NAMES)


class Model1DCNN(nn.Module):
    """
    1D-CNN architecture optimized for sequential spectral Raman profile vectors.
    Exact architecture from scripts/profiles 1fps.ipynb & profiles 30fps.ipynb.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=3)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=32)
        self.fnn1 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, sequence_length) -> unsqueeze to (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.batch_norm_1(x)
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.layer_norm_1(x)
        x = self.fnn1(x)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for dense spectral feature classification.
    Exact architecture from scripts/profiles 1fps.ipynb & profiles 30fps.ipynb.
    """
    def __init__(self, input_size: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def get_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = False):
    """
    ResNet50 backbone with replaced final FC linear head.
    Exact architecture from scripts/images 1fps.ipynb & images 30fps.ipynb.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_mobilevit(num_classes: int = NUM_CLASSES, pretrained: bool = False):
    """
    MobileViT-XS vision transformer for lightweight edge mineral classification.
    Exact architecture from scripts/images 1fps.ipynb & images 30fps.ipynb.
    """
    import timm
    model = timm.create_model('mobilevit_xs', pretrained=pretrained, num_classes=num_classes)
    return model
