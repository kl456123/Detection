# -*- coding: utf-8 -*-

from core.sampler import Sampler
import torch


class HardNegativeSampler(Sampler):
    def _subsample(self, mask, pos_mask, scores):
        """
        Hard Negative Mining
        """
        num_samples = self.num_samples

        # subsample from fg
        pos_scores = scores[pos_mask]
        pos_sorted_scores, pos_order = torch.sort(pos_scores, descending=False)
        fg_num = pos_scores.numel()
        if self.fg_num > fg_num:
            # subsample all pos
            mb_fg_inds = pos_order
        else:
            mb_fg_inds = pos_order[:self.fg_num]

        # the remain  is bg
        self.bg_num = num_samples - mb_fg_inds.numel()
        # subsample from bg
        neg_mask = mask and not pos_mask
        neg_scores = scores[neg_mask]
        neg_sorted_scores, neg_order = torch.sort(neg_scores, descending=True)
        bg_num = neg_scores.numel()
        if self.bg_num > bg_num:
            mb_bg_inds = neg_order
        else:
            mb_bg_inds = neg_order[:self.bg_num]

        # if not enough samples,oversample from fg
        if mb_fg_inds.numel() + mb_bg_inds.numel() < num_samples:
            pass
        mb_pos_mask = torch.zeros_like(pos_mask)
        mb_pos_mask[mb_fg_inds] = 1
        mb_neg_mask = torch.zeros_like(pos_mask)
        mb_neg_mask[mb_bg_inds] = 1

        mb_mask = mb_pos_mask and mb_neg_mask

        return mb_mask, mb_pos_mask
