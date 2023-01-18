import os
import json
import copy
import math
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

from .tools import get_mask_from_lengths, pad

from .blocks import (
    Mish,
    FCBlock,
    Conv1DBlock,
    SALNFFTBlock,
    MultiHeadAttention,
)


# from text.symbols import symbols

class MelStyleEncoder(nn.Module):
    """ Mel-Style Encoder """

    def __init__(
            self,
            n_mel_channels: int = 80,
            d_melencoder: int = 128,
            n_spectral_layer: int = 2,
            n_temporal_layer: int = 2,
            n_slf_attn_layer: int = 1,
            n_slf_attn_head: int = 2,
            conv_kernel_size: int = 5,
            encoder_dropout: int = 0.1,
    ):

        super(MelStyleEncoder, self).__init__()

        d_k = d_v = (
                d_melencoder // n_slf_attn_head
        )

        self.kernel_size = conv_kernel_size
        self.dropout = encoder_dropout

        self.fc_1 = FCBlock(n_mel_channels, d_melencoder)

        self.spectral_stack = nn.ModuleList(
            [
                FCBlock(
                    d_melencoder, d_melencoder, activation=Mish()
                )
                for _ in range(n_spectral_layer)
            ]
        )

        self.temporal_stack = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(
                        d_melencoder, 2 * d_melencoder, self.kernel_size, activation=Mish(), dropout=self.dropout
                    ),
                    nn.GLU(),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        # self.slf_attn_stack = nn.ModuleList(
        #     [
        #         MultiHeadAttention(
        #             n_slf_attn_head, d_melencoder, d_k, d_v, dropout=self.dropout, layer_norm=True
        #         )
        #         for _ in range(n_slf_attn_layer)
        #     ]
        # )
        #
        # self.fc_2 = FCBlock(d_melencoder, d_melencoder)

    def forward(self, mel, mask):

        # max_len = mel.shape[1]
        # slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1).to(mel.device)

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = residual + enc_output

        # Multi-head self-attention
        # for _, layer in enumerate(self.slf_attn_stack):
        #     residual = enc_output
        #     enc_output, _ = layer(
        #         enc_output, enc_output, enc_output, mask=slf_attn_mask
        #     )
        #     enc_output = residual + enc_output
        #
        # # Final Layer
        # enc_output = self.fc_2(enc_output)  # [B, T, H]
        #
        # # Temporal Average Pooling
        # enc_output = torch.mean(enc_output, dim=1, keepdim=True)  # [B, 1, H]

        return enc_output