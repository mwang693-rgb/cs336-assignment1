import torch
from torch import nn,Tensor
from .multihead_self_attention import multihead_self_attention
from .RMSNorm import RMSNorm
from .SwiGLU import SwiGLU

class Transformerblock(nn.Module):
    def __init__(self,d_model : int,
                num_head: int,
                d_ff: int,
                max_seq_len: int,
                theta: int):
        super().__init__()
        self.RMSNorm1 = RMSNorm(d_model)
        self.RMSNorm2 = RMSNorm(d_model)
        self.multilayer = multihead_self_attention(d_model,num_head,theta,max_seq_len)
        self.FFNlayer = SwiGLU(d_model,d_ff)
    def forward(self,x:Tensor) -> Tensor:
        x = x + self.multilayer(self.RMSNorm1(x))
        return x + self.FFNlayer(self.RMSNorm2(x))