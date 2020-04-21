import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.config import cfg

class Distillation(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
    
        self.distillation = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.distillation.weight, std=0.01)
        nn.init.constant_(self.distillation.bias, 0)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        
        distillation = F.softmax(self.distillation(x), dim=1)

        return distillation
