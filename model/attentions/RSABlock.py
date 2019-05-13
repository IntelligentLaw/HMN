import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .AoA import AoA
class RSABlock(nn.Module):
    def __init__(self):
        super(RSABlock, self).__init__()
        self.AoALayer = AoA()

    def forward(self, source_input, source_lengths, target_input, target_lengths):
        """
        :param source_input: (B, L, H)
        :param source_lengths: (B, 1)
        :param target_input: (B, LS, H)
        :param target_lengths: (B, 1)
        :return:ret_source
        :return:ret_target
        """
        source_raw = source_input
        target_raw = target_input

        avg_alpha, avg_beta = self.AoALayer.forward(source_input, source_lengths, target_input, target_lengths)

        ret_source = source_raw + source_raw * avg_alpha
        ret_target = target_raw + target_raw * avg_beta
        # ret_source = source_raw * avg_alpha
        # ret_target = target_raw * avg_beta

        return ret_source, ret_target

