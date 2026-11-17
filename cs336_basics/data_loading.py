import numpy as np
import torch
from torch import Tensor
def data_loading(
        x:np.ndarray,
        batch_size: int,
        context_length: int,
        device
):

    inputs = []
    targets = []
    starts = np.random.randint(
    0,
    len(x) - context_length,
    size=batch_size,
)

    for start in starts:
        inputs.append(x[start : start + context_length])
        targets.append(x[start + 1 : start + context_length + 1])

    inputs = np.stack(inputs)
    targets = np.stack(targets)

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)


    return inputs,targets

