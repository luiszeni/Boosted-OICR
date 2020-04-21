import torch
import torch.nn as nn
from pdb import set_trace as pause 

def weighted_clusters_softmax_loss(pcl_probs, labels, cls_loss_weights, gt_assignment,
                pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                im_labels):

        eps = 1e-6
        # pause()
        bkg_loss = pcl_probs[:, 0]
        bkg_loss = bkg_loss.clamp(eps, 1-eps).log() * cls_loss_weights.cuda()

        bkg_loss = bkg_loss[labels == 0]
        bkg_loss = bkg_loss.sum()

        clus_loss = pc_probs.clamp(eps, 1-eps).cuda().log() * img_cls_loss_weights
        clus_loss = clus_loss.sum()

        loss = (-(bkg_loss + clus_loss)) / pcl_probs.size(0)

        return loss

