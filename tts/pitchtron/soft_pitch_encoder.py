# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Style encoder of GST-Tacotron."""

from typing import Sequence

import torch
import logging

from typing import Optional
from nets.transformer.attention import (
    MultiHeadedAttention as BaseMultiHeadedAttention,  # NOQA
)


class SoftPitchEncoder(torch.nn.Module):
    def __init__(
            self,
            idim: int = 80,
            sp_token_dim: int = 256,  # 384
            sp_heads: int = 4,  # 4
            conv_layers: int = 6,
            conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
            conv_kernel_size: int = 3,
            conv_stride: int = 2,
            gru_layers: int = 1,
            gru_units: int = 128,  # 128 from ref wav
    ):
        """Initilize global style encoder module."""
        super(SoftPitchEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            sp_token_dim=sp_token_dim,
            sp_heads=sp_heads,
        )
        self.gru_units = gru_units

    def forward(self,
                f0: torch.Tensor = None,
                text_embs: torch.Tensor = None,
                gst_index: int = None,
                ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        """
        if f0 is not None:
            # f0 = f0.unsqueeze(dim=-1)
            ref_embs = self.ref_enc(f0)
            # logging.info(f'ref_embs: {ref_embs.shape}')
        else:
            ref_embs = torch.zeros(1, self.gru_units)

        # style_embs = self.stl(ref_embs, gst_mask)
        style_embs = self.stl(ref_embs, text_embs, gst_index)

        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.

    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    """

    def __init__(
            self,
            idim=80,
            conv_layers: int = 6,
            conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
            conv_kernel_size: int = 3,
            conv_stride: int = 2,
            gru_layers: int = 1,
            gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
                len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv1d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm1d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                                   gru_in_units - conv_kernel_size + 2 * padding
                           ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(128, gru_units, gru_layers, batch_first=True)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = f0.size(0)
        xs = f0.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    """

    def __init__(
            self,
            ref_embed_dim: int = 128,
            sp_token_dim: int = 256,
            sp_heads: int = 4,
            dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=sp_token_dim,
            v_dim=sp_token_dim,
            n_head=sp_heads,
            n_feat=sp_token_dim,
            dropout_rate=dropout_rate,
        )
        self.heads = sp_heads

    def forward(self,
                ref_embs: torch.Tensor,
                text_embs: torch.Tensor,
                gst_index: [torch.Tensor, int] = None,
                ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        # gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)

        if gst_index is not None:
            gst_mask = torch.zeros(1, self.gst_heads, 1, self.gst_tokens)
            for i in range(self.gst_heads):
                gst_mask[0][i][0][gst_index] = 1
        else:
            gst_mask = None

        style_embs = self.mha(ref_embs, text_embs, text_embs, gst_mask)

        return style_embs.squeeze(1)


class MultiHeadedAttention(BaseMultiHeadedAttention):
    """Multi head attention module with different input dimension."""

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # NOTE(kan-bayashi): Do not use super().__init__() here since we want to
        #   overwrite BaseMultiHeadedAttention.__init__() method.
        torch.nn.Module.__init__(self)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
