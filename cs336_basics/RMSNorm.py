import torch
from torch import nn, Tensor
class RMSNorm(nn.Module):
    def __init__(self, d_model:int,eps:float = 1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight =  nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        x_normed = x_float / torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        result = x_normed*self.weight
        return result.to(in_dtype)
        