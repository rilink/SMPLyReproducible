"""
CNN-HAR on acc+gyr windows.
Input: 6 channels (acc_x,y,z, gyr_x,y,z), 150-frame windows at 50 Hz.
Architecture: Conv2D(1->50, 5x1) -> Conv2D(50->40, 5x1) -> Conv2D(40->20, 3x1)
  -> Conv2D(20->400, 1xchannels) -> AdaptiveMaxPool -> Linear(400, 8).
"""

import argparse

import torch
from torch import nn

import common


class CNNHAR(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(5, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1))
        self.norm1 = nn.LocalResponseNorm(5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        self.conv2 = nn.Conv2d(50, 40, kernel_size=(5, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1))
        self.norm2 = nn.LocalResponseNorm(5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        self.conv3 = nn.Conv2d(40, 20, kernel_size=(3, 1))
        self.norm3 = nn.LocalResponseNorm(5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        self.conv4 = nn.Conv2d(20, 400, kernel_size=(1, num_channels))
        self.norm4 = nn.LocalResponseNorm(5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        self.activation = nn.ReLU(inplace=True)
        self.temporal_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Conv2d(400, num_classes, kernel_size=(1, 1))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x = self.norm1(self.pool1(self.activation(self.conv1(x))))
        x = self.norm2(self.pool2(self.activation(self.conv2(x))))
        x = self.norm3(self.activation(self.conv3(x)))
        x = self.norm4(self.activation(self.conv4(x)))
        return self.temporal_pool(x).squeeze(-1).squeeze(-1)  # (B, 400)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_embeddings(x)
        return self.classifier(z.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)


def make_model():
    return CNNHAR(num_channels=common.NUM_CHANNELS, num_classes=common.NUM_CLASSES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", default="forearm", choices=["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    common.run_cv(make_model, model_tag="cnnhar", position=args.position, epochs=args.epochs)
