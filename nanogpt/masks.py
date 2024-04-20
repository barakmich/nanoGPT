import torch

def create_attn_mask(x, null_token_id):
    x = x.ne(null_token_id).to(torch.float32)
    #x = x.where(x != 0.0, -torch.inf)
    return x
