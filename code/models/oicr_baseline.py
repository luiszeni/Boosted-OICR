import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.config import cfg

from models.vgg16                import VGG16Backbone
from layers.roi_pooling import RoiPoolLayer
from layers.refinement.oicr      import OICR             as Refinement
from layers.loss.mil_loss        import mil_loss
from layers.refinement_agents    import RefinementAgents
from layers.mil                  import MIL


import logging
logger = logging.getLogger(__name__)

from pdb import set_trace as pause

class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_iter = 0 
       
        self.backbone          = VGG16Backbone()
        self.box_features      = RoiPoolLayer(self.backbone.dim_out, self.backbone.spatial_scale)
        self.mil               = MIL(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES)
        self.refinement_agents = RefinementAgents(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES + 1)


    def set_inner_iter(self, inner_iter):
        self.inner_iter = inner_iter 

    def forward(self, im_data, rois, labels):

        with torch.set_grad_enabled(self.training):
            rois   = rois.squeeze(dim=0)
            labels = labels.squeeze(dim=0)

            backbone_feat = self.backbone(im_data)
            box_feat      = self.box_features(backbone_feat, rois)
            mil_score     = self.mil(box_feat)
            refine_score  = self.refinement_agents(box_feat)


            im_cls_score = mil_score.sum(dim=0, keepdim=True)


            return_dict = {} 
            if self.training:
                return_dict['losses'] = {}

                # image classification loss
                loss_im_cls = mil_loss(im_cls_score, labels)
                return_dict['losses']['loss_im_cls'] = loss_im_cls

                # refinement losses
                for i_refine, refine in enumerate(refine_score):
                    if i_refine == 0:
                        refine_loss = Refinement(rois[:, 1:], mil_score, labels, refine)
                    else:
                        refine_loss = Refinement(rois[:, 1:], refine_score[i_refine - 1], labels, refine)

                    return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()
                    
            else:
                return_dict['blob_conv'] = backbone_feat
                return_dict['rois'] = rois
                return_dict['cls_score'] = im_cls_score
                return_dict['refine_score'] = refine_score

            return return_dict

def loot_model(args):
    print("Using model description:", args.model)
    model = DetectionModel()

    if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        model.backbone.load_pretrained_imagenet_weights(model)

    return model


