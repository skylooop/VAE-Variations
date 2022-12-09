import torch.nn as nn
import torch
from torch import Tensor
import torch.functional as F
from torch.distributions import OneHotCategorical


def sample_gumble(size: tuple(int, int), eps: float = 1e-20) -> Tensor:
    '''
    Finding samples from Gumble distribution
    Sample U_k ~ U(0, 1) and compute -log(-log(U_k))
    '''
    U: torch.Tensor = torch.rand(size).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_samples(logits, thau: float = 3.0) -> Tensor:
    y = logits + sample_gumble(logits.shape())
    return F.softmax(y / thau, dim = -1)

def gumbel_softmax(logits, temperature, cfg, evaluate=False):
    if evaluate:
        d = OneHotCategorical(logits.view(-1, cfg.latent_dim, cfg.categorical_dim))
        return d.sample().view(cfg.latent_dim * cfg.categorical_dim)

    y = gumbel_softmax_samples(logits, temperature)
    return y.view(-1, cfg.latent_dim * cfg.categorical_dim)
