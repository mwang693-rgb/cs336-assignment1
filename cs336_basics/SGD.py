from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                #p.data -= lr * grad
                state["t"] = t + 1  # Increment iteration number.
        return loss


if __name__ == "__main__":
    torch.manual_seed(42)
    initial_weights = 5 * torch.randn((10, 10))
    for lr in (10, 100, 1000):
        weights = torch.nn.Parameter(initial_weights.clone())
        opt = SGD([weights], lr=lr)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = weights.square().mean()
            losses.append(loss.item())
            loss.backward()
            opt.step()
        print(f"lr={lr}: {losses}")
