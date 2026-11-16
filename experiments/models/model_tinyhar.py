"""
TinyHAR on acc+gyr windows.
Input: 6 channels (acc_x,y,z, gyr_x,y,z), 150-frame windows at 50 Hz.
Architecture: 4x Conv2D(64) channel-wise -> per-timestep SelfAttention over
  channels -> cross-channel FC fusion -> GRU -> temporal FC fusion -> Linear(128, 8).
TinyHAR_Model is built with the SelfAttention_interaction + FC path. An
alternative Transformer-based channel-interaction path is unused and not
included here, along with its einops dependency.
"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn

import common


class SelfAttentionInteraction(nn.Module):
    """Attention across the channel (sensor) dimension, applied per timestep."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, sensor_channel, n_channels)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)
        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        return o.permute(0, 2, 1).contiguous()


class TemporalGRU(nn.Module):
    def __init__(self, filter_num: int):
        super().__init__()
        self.rnn = nn.GRU(filter_num, filter_num, 1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return outputs


class TinyHAR(nn.Module):
    """Input: (B, 1, T, C). Embedding: (B, 2*filter_num)."""

    def __init__(self, input_shape: tuple[int, int, int, int], num_classes: int, filter_num: int = 64):
        super().__init__()
        filter_size_list = [5, 5, 3, 3]
        filter_nums = [1] + [filter_num] * 4

        self.layers_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(filter_nums[i], filter_nums[i + 1], (filter_size_list[i], 1),
                           stride=(2, 1) if i % 2 == 0 else (1, 1)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_nums[i + 1]),
            )
            for i in range(4)
        ])

        num_channels = input_shape[3]
        downsampling_length = self._trace_output_length(input_shape)

        self.channel_interaction = SelfAttentionInteraction(filter_num)
        self.channel_fusion = nn.Linear(num_channels * filter_num, 2 * filter_num)
        self.activation = nn.ReLU()
        self.temporal_interaction = TemporalGRU(2 * filter_num)
        self.dropout = nn.Dropout(0.1)
        self.temporal_fusion = nn.Linear(downsampling_length * 2 * filter_num, 2 * filter_num)
        self.prediction = nn.Linear(2 * filter_num, num_classes)

    def _trace_output_length(self, input_shape) -> int:
        x = torch.rand(input_shape)
        for layer in self.layers_conv:
            x = layer(x)
        return x.shape[2]

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, C)
        for layer in self.layers_conv:
            x = layer(x)
        x = x.permute(0, 3, 2, 1)  # (B, F, T', C) -> (B, C, T', F)

        # Channel-interaction weights are shared across timesteps and each timestep
        # is attended independently, so batch B and T' together into one call
        # instead of looping T' times -- same result, far fewer kernel launches.
        b, c, t_len, f = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(b * t_len, c, f)  # (B*T', C, F)
        x = self.channel_interaction(x_flat).reshape(b, t_len, c, f).permute(0, 2, 3, 1)  # (B, C, F, T')
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2).reshape(x.shape[0], x.shape[3], -1)  # (B, T', C*F)
        x = self.activation(self.channel_fusion(x))  # (B, T', 2F)

        x = self.temporal_interaction(x)  # (B, T', 2F)
        x = x.reshape(x.shape[0], -1)  # (B, T'*2F)
        return self.activation(self.temporal_fusion(x))  # (B, 2F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(self.get_embeddings(x))


class TinyHARWrapper(nn.Module):
    """Adapter: accepts (B, T, C), forwards to TinyHAR as (B, 1, T, C)."""

    def __init__(self, window_size: int, num_channels: int, num_classes: int, filter_num: int = 64):
        super().__init__()
        self._core = TinyHAR(
            input_shape=(1, 1, window_size, num_channels), num_classes=num_classes, filter_num=filter_num,
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._core.get_embeddings(x.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x.unsqueeze(1))


def make_model():
    return TinyHARWrapper(window_size=common.WINDOW_SIZE, num_channels=common.NUM_CHANNELS, num_classes=common.NUM_CLASSES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", default="forearm", choices=["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    common.run_cv(make_model, model_tag="tinyhar", position=args.position, epochs=args.epochs)
