import torch
import torch.nn as nn
from pdb import set_trace as pause 

class OICRLosses(nn.Module):

    def forward(self, pcl_probs, labels, cls_loss_weights, gt_assignment, im_labels):


        eps = 1e-6
        pcl_probs = pcl_probs.clamp(eps, 1-eps).log()

        
        cls_loss_weights = cls_loss_weights.repeat(pcl_probs.shape[1],1).permute(1,0).cuda()


        labels = labels.repeat(pcl_probs.shape[1],1).permute(1,0).long()
        reap   = torch.arange(pcl_probs.shape[1])[None,:].repeat(pcl_probs.shape[0], 1).long()
        labels = (reap - labels == 0).float().cuda()

        
        loss = labels * cls_loss_weights * pcl_probs

        loss = -loss.sum(dim=0).sum() / pcl_probs.size(0)


        return loss

