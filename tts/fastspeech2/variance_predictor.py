#!/usr/bin/env python3

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Variance predictor related modules."""

import torch

# from typeguard import check_argument_types

from nets.transformer.layer_norm import LayerNorm
from tts.meta_style.blocks import StyleAdaptiveLayerNorm


class VariancePredictor(torch.nn.Module):
    """Variance predictor module.

    This is a module of variacne predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        idim: int,
        n_layers: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
        spectrogram: bool = False,
        use_equalizer: bool = False,
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int): Number of convolutional layers.
            n_chans (int): Number of channels of convolutional layers.
            kernel_size (int): Kernel size of convolutional layers.
            dropout_rate (float): Dropout rate.

        """
        # assert check_argument_types()
        super().__init__()
        self.conv = torch.nn.ModuleList()

        if use_equalizer:
            for idx in range(n_layers):
                in_chans = idim if idx == 0 else n_chans
                self.conv += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_chans,
                            n_chans,
                            kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                            bias=bias,
                        ),
                        torch.nn.ReLU(),
                        StyleAdaptiveLayerNorm(384, n_chans),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
        else:
            for idx in range(n_layers):
                in_chans = idim if idx == 0 else n_chans
                self.conv += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_chans,
                            n_chans,
                            kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                            bias=bias,
                        ),
                        torch.nn.ReLU(),
                        LayerNorm(n_chans, dim=1),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]

        if spectrogram:
            self.linear = torch.nn.Linear(n_chans, 10)
        else:
            self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor = None, style: bool=None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).

        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        if style is not None:
            for f in self.conv:
                for i, _f in enumerate(f):
                    if i == 2:
                        xs = _f(xs, style)
                    else:
                        xs = _f(xs)  # (B, C, Tmax)
        else:
            for f in self.conv:
                xs = f(xs)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs
