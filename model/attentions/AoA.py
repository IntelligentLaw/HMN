import torch.nn as nn
import torch.nn.functional as F
import torch
import math
def create_mask(seq_lens):
    mask = torch.zeros(len(seq_lens), torch.max(seq_lens))
    for i, seq_len in enumerate(seq_lens):
        mask[i][:seq_len] = 1
    return mask.float()

def softmax_mask(input, mask, axis=1, epsilon=1e-12):
    """
    :param input: (B, LS, L)
    :param mask: (B, LS, L)
    :param axis: 1
    :return:
    """
    # (B, LS, L) -> (B, LS)
    shift, _ = torch.max(input, axis, keepdim=True)
    shift = shift.expand_as(input)#.cuda()
    target_exp = torch.exp(input - shift) * mask
    normalize = torch.sum(target_exp, axis, keepdim=True).expand_as(target_exp)
    softm = target_exp / (normalize + epsilon)

    return softm#.cuda()



class AoA(nn.Module):
    def __init__(self):
        super(AoA, self).__init__()

    def forward(self, source_input, source_lengths, target_input, target_lengths):
        """
        :param source_input: (B, L, H)
        :param source_lengths: (B, 1)
        :param target_input: (B, LS, H)
        :param target_lengths: (B, 1)
        :return: (B, L, H), (B, LS, H)
        """
        ###### 1. BEGIN: create mask
        # (B, 1)->(B, L)
        source_mask = create_mask(source_lengths).cuda()
        # (B, L) -> (B, L, 1)
        source_mask = source_mask.unsqueeze(2)
        # (B, 1)->(B, LS)
        target_mask = create_mask(target_lengths).cuda()
        # (B, LS) -> (B, LS, 1)
        target_mask = target_mask.unsqueeze(2)
        ###### 1. END: create mask


        # (B, L, H) -> (B, H, L)
        source_input = torch.transpose(source_input, 1, 2)

        ###### BEGIN: pair-wise matching score
        # (B, LS, H) * (B, H, L) -> (B, LS, L)
        M = torch.bmm(target_input, source_input)
        # (B, LS, 1) * (B, 1, L) -> (B, LS, L)
        M_mask = torch.bmm(target_mask, source_mask.transpose(1, 2))
        ###### END: pair-wise matching score

        ###### BEGIN: Column Wise Softmax and Row Wise Softmax
        # (B, LS, L), (B, LS, L) -> (B, LS, L)
        alpha = softmax_mask(M, M_mask, axis=1)
        # (B, LS, L), (B, LS, L) -> (B, LS, L)
        beta = softmax_mask(M, M_mask, axis=2)
        ###### END: Column Wise Softmax and Row Wise Softmax


        # (B, LS, L) -> (B, 1, L)
        sum_beta = torch.sum(beta, dim=1, keepdim=True)
        # (B) -> (B, 1) -> (B, 1, 1) -> (B, 1, L)
        docs_len = target_lengths.unsqueeze(1).unsqueeze(2).expand_as(sum_beta)
        # (B, 1, L)/(B, 1, L) -> (B, 1, L)
        average_beta = sum_beta/docs_len.float()
        # (B, 1, L) -> (B, L) ->(B, L, 1) -> (B, L, H)
        squeeze_beta = average_beta.squeeze(1).unsqueeze(-1).repeat((1, 1, source_input.size(1)))


        # (B, LS, L) -> (B, LS, 1)
        sum_alpha = torch.sum(alpha, dim=2, keepdim=True)
        # (B) -> (B, 1) -> (B, 1, 1) -> (B, LS, 1)
        query_len = source_lengths.unsqueeze(1).unsqueeze(2).expand_as(sum_alpha)
        # (B, LS, 1)/(B, LS, 1) -> (B, LS, 1)
        average_alpha = sum_alpha/query_len.float()
        # (B, LS, 1) -> (B, LS, H)
        squeeze_alpha = average_alpha.repeat((1, 1, target_input.size(-1)))



        return squeeze_beta, squeeze_alpha
