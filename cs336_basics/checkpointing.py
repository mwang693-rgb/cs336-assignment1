import os
import torch
from torch import nn, optim
import typing
def save_checkpoint(
        model: torch.nn.Module,  
        optimizer: torch.optim.Optimizer,  
        iteration: int,  
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
        checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                }
        torch.save(checkpoint, out)
        return out

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    param = next(model.parameters(), None)
    device = param.device if param is not None else None
    checkpoint = torch.load(src, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]