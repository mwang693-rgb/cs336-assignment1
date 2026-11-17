import torch
from torch import Tensor


def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(dim=-1, index=target.long().unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()
