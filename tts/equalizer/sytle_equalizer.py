import torch
import torch.nn as nn
import torch.nn.functional as F
from ..meta_style.blocks import (
    Mish,
    FCBlock,
    Conv1DBlock,
    MultiHeadAttention,
)


class StyleEqualizer(nn.Module):
    def __init__(
            self,
            input_size: int = 384,
            output_size: list = [192, 64],
            bias=False,
            attn_head: int = 4,
            input_d: int = 384,
            dropout: float = 0.2,
    ):
        super(StyleEqualizer, self).__init__()

        self.linear_layer_1 = nn.Linear(input_size, output_size[0], bias=bias)
        self.linear_layer_2 = nn.Linear(output_size[0], output_size[1], bias=bias)

        d_k = d_v = input_d // attn_head

        # self-attention
        # self.final_linear = nn.Linear(input_size, input_size)

        # attention
        self.attention_linear = nn.Linear(input_size, input_size)
        self.final_linear = nn.Linear(input_size, input_size)

        self.multi_head_attention = MultiHeadAttention(attn_head, input_d, d_k, d_v, dropout=dropout, layer_norm=False)

    def forward(self, src_style, ref_style, ref_masks, hs):
        res_ref_style = ref_style
        max_len = ref_masks.shape[1]
        slf_attn_mask = ref_masks.unsqueeze(1).expand(-1, max_len, -1).to(ref_style.device)
        src_style = self.linear_layer_1(src_style)
        src_style = self.linear_layer_2(src_style)

        ref_style = self.linear_layer_1(ref_style)
        ref_style = self.linear_layer_2(ref_style)

        difference = torch.mean(src_style, dim=1) - torch.mean(ref_style, dim=1)

        style = res_ref_style + torch.matmul(torch.matmul(difference, self.linear_layer_2.weight), self.linear_layer_1.weight).unsqueeze(1)

        # self-attention
        # residual = style
        # style, _ = self.multi_head_attention(style, style, style, slf_attn_mask)
        # style = style + residual
        #
        # style = torch.mean(self.final_linear(style), dim=1, keepdim=True)

        # attention
        style = torch.mean(self.attention_linear(style), dim=1, keepdim=True)
        style = style.expand(hs.size())
        style, _ = self.multi_head_attention(hs, style, style)
        style = torch.mean(self.final_linear(style), dim=1, keepdim=True)

        return style


# def average_mel(mels, ds):
#     averaged_mel = list()
#     averaged_mels = list()
#
#     for d, mel in zip(ds, mels):
#         d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
#
#         temp_mel = list()
#
#         for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
#             if start == end:
#                 continue
#             temp_mel.append()
#
#
#     return averaged_mel