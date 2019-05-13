import torch

def get_trainable_param_num(model):
    """ get the number of trainable parameters

    Args:
        model:

    Returns:

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_param_num(model):
    """ get the number of parameters

    Args:
        model:

    Returns:

    """
    return sum(p.numel() for p in model.parameters())



def create_mask(seq_lens):
    mask = torch.zeros(len(seq_lens), torch.max(seq_lens))
    for i, seq_len in enumerate(seq_lens):
        mask[i][:seq_len] = 1

    return mask.float()

