"""
DeepConvLSTM on acc+gyr windows.
Input: 6 channels (acc_x,y,z, gyr_x,y,z), 150-frame windows at 50 Hz.
Architecture: 4x Conv2D(64, k=(5,1)) -> LSTM(128) -> Linear(128, 8).
"""

import argparse

import torch
from torch import nn

import common


class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        conv_kernels: int = 64,
        conv_kernel_size: int = 5,
        lstm_units: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.activation = nn.ReLU()

        lstm_input_size = conv_kernels * channels
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(0, 2, 1, 3)  # (B, T', kernels, C)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, T', kernels*C)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]  # (B, lstm_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.get_embeddings(x)))


def make_model():
    return DeepConvLSTM(channels=common.NUM_CHANNELS, classes=common.NUM_CLASSES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", default="forearm", choices=["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    common.run_cv(make_model, model_tag="deepconvlstm", position=args.position, epochs=args.epochs)
