import math
def get_lr_cosine_schedule(
        it: int,
        max: float,
        min: float,
        warmup_iters:int,
        cosine_cycle_iters:int
) -> float:
    if it < warmup_iters:
        return max*it/warmup_iters
    if it >= cosine_cycle_iters:
        return min
    decay_ratio = (it - warmup_iters ) / (cosine_cycle_iters - warmup_iters)

    coeff = 0.5 *(1 + math.cos(math.pi * decay_ratio ))

    return min + coeff*(max-min)