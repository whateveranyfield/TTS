import torch
import pycwt as wavelet
import numpy as np
from sklearn import preprocessing


class stats_predictor(torch.nn.Module):
    def __init__(self, adim=385, pitch_embed_kernel_size=5):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=adim,
            out_channels=adim,
            kernel_size=pitch_embed_kernel_size,
            padding=(pitch_embed_kernel_size - 1) // 2,
        )

        self.linear = torch.nn.Linear(adim, 1)

    def forward(self, x):
        x = self.conv(x)

        # average to global vector
        x = torch.mean(x, dim=2)

        return self.linear(x)


# def inverse_cwt_torch(Wavelet_lf0, scales):
#
#     b = ((torch.arange(0, len(scales)).float().to(Wavelet_lf0.device)[None, None, :] + 1 + 2.5) ** (-2.5))
#     lf0_rec = Wavelet_lf0 * b
#     lf0_rec_sum = lf0_rec.sum(-1)
#     lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
#
#     return lf0_rec_sum


def inverse_cwt(wavelet_lf0, scales=None, device="cuda"):

    if scales is None:
        length = 10
    else:
        length = scales.shape[1]

    lf0_rec = torch.zeros(wavelet_lf0.shape[0], wavelet_lf0.shape[1], length, device=device)

    for i in range(0, length):
        c = torch.mul(wavelet_lf0[:, :, i], torch.full((wavelet_lf0.shape[0], 1), (i + 1 + 2.5) ** (-2.5)).to(device))
        lf0_rec[:, :, i] = torch.squeeze(c)

    lf0_rec_sum = lf0_rec.sum(dim=2)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
    # lf0_rec_sum = torch.from_numpy(preprocessing.scale(lf0_rec_sum.detach().cpu().numpy(), axis=1)).to(device)

    return lf0_rec_sum