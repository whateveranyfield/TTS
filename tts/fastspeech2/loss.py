# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple

import torch
import torch.nn.functional as F

# from typeguard import check_argument_types

from nets.fastspeech.duration_predictor import (
    DurationPredictorLoss,  # noqa: H301
)
from nets.nets_utils import make_non_pad_mask
# from rmi import RMILoss


class FastSpeech2Loss(torch.nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(self,
                 use_masking: bool = True,
                 use_weighted_masking: bool = False,
                 use_cwt: bool = False,
                 use_classifier: bool = False,
                 use_mean_loss: bool = False,
                 use_pitch_max: bool = False,
                 use_equalizer: bool = False,
                 ):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        # assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.use_cwt = use_cwt
        self.use_classifier = use_classifier
        self.use_mean_loss = use_mean_loss
        self.use_pitch_max = use_pitch_max
        self.use_equalizer = use_equalizer

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        # self.mutual_information = RMILoss(with_logits=True)

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        p_m_outs: torch.Tensor = None,
        pms: torch.Tensor=None,
        cs_mean: torch.Tensor = None,
        cs_std: torch.Tensor = None,
        c_mean_outs: torch.Tensor = None,
        c_std_outs: torch.Tensor = None,
        p_outs_after: torch.Tensor = None,
        cs: torch.Tensor = None,
        ec_outs: torch.Tensor = None,
        sids: torch.Tensor = None,
        styles: torch.Tensor = None,
        hs_l: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part

        mean_lengths = torch.Tensor([1 for _ in range(ps.shape[0])])

        if self.use_mean_loss:
            ms = list()
            m_outs = list()

            for d, y, after in zip(ds, ys, after_outs):
                d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
                temp_ms = list()
                temp_m_outs = list()
                for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
                    temp_ms.append(y[start:end].mean())
                    temp_m_outs.append(after[start:end].mean())
                temp_ms = torch.stack(temp_ms)
                temp_m_outs = torch.stack(temp_m_outs)
                ms.append(temp_ms)
                m_outs.append(temp_m_outs)

            ms = torch.stack(ms).nan_to_num()
            m_outs = torch.stack(m_outs).nan_to_num()

            if self.use_masking:
                duration_masks = make_non_pad_mask(ilens).to(ys.device)
                m_outs = m_outs.masked_select(duration_masks)
                ms = ms.masked_select(duration_masks)

            mean_loss = self.mse_criterion(m_outs, ms)

        # if self.use_equalizer:
        #     a_styles = list()
        #
        #     for d, style in zip(ds, styles):
        #         d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        #         temp_style = list()
        #         for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
        #             temp_style.append(style[start:end].mean())
        #         temp_style = torch.stack(temp_style)
        #         a_styles.append(temp_style)
        #
        #     a_styles = torch.stack(a_styles).nan_to_num()
        #
        #     mi_loss = self.mutual_information(a_styles, hs_l)

        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

            if self.use_cwt:
                pitch_masks_stats = make_non_pad_mask(mean_lengths).unsqueeze(-1).to(ys.device)
                p_outs_after = p_outs_after.masked_select(pitch_masks)
                cs = cs.masked_select(pitch_masks)
                cs_std = cs_std.masked_select(pitch_masks_stats)
                cs_mean = cs_mean.masked_select(pitch_masks_stats)
                c_mean_outs = c_mean_outs.masked_select(pitch_masks_stats)
                c_std_outs = c_std_outs.masked_select(pitch_masks_stats)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)

        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        energy_loss = self.mse_criterion(e_outs, es)

        if self.use_cwt:
            cwt_mean_loss = self.mse_criterion(c_mean_outs, cs_mean)
            cwt_std_loss = self.mse_criterion(c_std_outs, cs_std)
            cwt_pitch_loss = self.mse_criterion(p_outs, cs)
            pitch_loss = self.mse_criterion(p_outs_after, ps)
        else:
            pitch_loss = self.mse_criterion(p_outs, ps)

        if self.use_pitch_max:
            pitch_max_loss = self.mse_criterion(p_m_outs, pms)

        if self.use_classifier:
            classifier_loss = self.cross_entropy(ec_outs, sids)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        if self.use_mean_loss:
            if self.use_cwt and self.use_classifier and self.use_pitch_max:
                return l1_loss, duration_loss, cwt_pitch_loss, energy_loss, cwt_mean_loss, cwt_std_loss, classifier_loss, pitch_max_loss, mean_loss
            elif self.use_cwt and self.use_pitch_max:
                return l1_loss, duration_loss, cwt_pitch_loss, energy_loss, cwt_mean_loss, cwt_std_loss, pitch_max_loss, mean_loss
            elif self.use_cwt:
                return l1_loss, duration_loss, cwt_pitch_loss, energy_loss, cwt_mean_loss, cwt_std_loss, mean_loss
            elif self.use_pitch_max:
                return l1_loss, duration_loss, energy_loss, pitch_max_loss, mean_loss
        else:
            if self.use_cwt and self.use_classifier and self.use_pitch_max:
                return l1_loss, duration_loss, cwt_pitch_loss, energy_loss, cwt_mean_loss, cwt_std_loss, classifier_loss, pitch_max_loss, pitch_loss
            elif self.use_cwt and self.use_pitch_max:
                return l1_loss, duration_loss, cwt_pitch_loss, energy_loss, cwt_mean_loss, cwt_std_loss, pitch_max_loss, pitch_loss
            elif self.use_cwt:
                return l1_loss, duration_loss, cwt_pitch_loss, energy_loss, cwt_mean_loss, cwt_std_loss, pitch_loss
            elif self.use_pitch_max:
                return l1_loss, duration_loss, pitch_loss, energy_loss, pitch_max_loss

        return l1_loss, duration_loss, pitch_loss, energy_loss
