from collections.abc import Iterable

import torch


def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    total_norm = 0.0
    for p in params_with_grad:
        params_norm = torch.norm(p.grad.detach(), p=2)
        total_norm += (params_norm.item() **2)
    total_norm = total_norm ** 0.5

    eps = 1e-6
    if total_norm > max_norm:
        clip = max_norm/(eps+total_norm)

        for p in params_with_grad:
            p.grad.detach().mul_(clip)

