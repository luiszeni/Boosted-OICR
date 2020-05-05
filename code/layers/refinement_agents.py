import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.config import cfg

class RefinementAgents(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.refine_score = []
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)

        self._init_weights()

    def _init_weights(self):
        for i_refine in range(cfg.REFINE_TIMES):
            nn.init.normal_(self.refine_score[i_refine].weight, std=0.01)
            nn.init.constant_(self.refine_score[i_refine].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        for i_refine in range(cfg.REFINE_TIMES):
            detectron_weight_mapping.update({
                'refine_score.%d.weight' % i_refine: 'refine_score%d_w' % i_refine,
                'refine_score.%d.bias' % i_refine: 'refine_score%d_b' % i_refine
            })
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        refine_score = [F.softmax(refine(x), dim=1) for refine in self.refine_score]

        return refine_score
