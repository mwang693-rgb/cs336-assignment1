import torch
from torch import nn, Tensor
from .Embedding import embedding
from .Transformerblock import Transformerblock
from .Linear import Linear
from .RMSNorm import RMSNorm
class Transformer_lm(nn.Module):
    def __init__(self,d_model : int,
                 num_head: int,
                 d_ff: int,
                 vocab_size:int, 
                 context_length:int,
                 num_layer:int,
                 rope_theta:int):
            super().__init__()
            self.Embedding = embedding(vocab_size,d_model)
            self.layer = nn.ModuleList([Transformerblock(d_model,num_head,d_ff,context_length,rope_theta) for _ in range(num_layer)])
            self.Norm = RMSNorm(d_model)
            self.lm_head = Linear(d_model, vocab_size)
            self.context_length = context_length
    def forward(self, sentence : Tensor) -> Tensor:
        x = self.Embedding(sentence)
        for block in self.layer:
            x = block(x)
        x = self.Norm(x)
        return self.lm_head(x)
    
    


        

        
        

    

                                         
                                       