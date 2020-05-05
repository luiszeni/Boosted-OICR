import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.config import cfg

from models.vgg16  import VGG16Backbone

import logging
logger = logging.getLogger(__name__)

from pdb import set_trace as pause


import utils.vgg_weights_helper as vgg_utils

from layers.refinement.oicr            import OICR         as Refinement
from layers.losses.oicr_losses         import OICRLosses   as Losses
from layers.losses.mil_loss            import mil_loss
from layers.refinement_agents          import RefinementAgents
from layers.distillation               import Distillation
from layers.mil                        import MIL


from layers.roi_pooling.roi_pool import RoiPoolLayer
from layers.adaptative_supervision_functions import get_adaptative_lambda

class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
       
        # Backbone for feature extraction
        self.backbone = VGG16Backbone()


        self.box_features      = RoiPoolLayer(self.backbone.dim_out, self.backbone.spatial_scale)
        self.mil               = MIL(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES)
        self.refinement_agents = RefinementAgents(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES + 1)
        self.distillation      = Distillation(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Refine_Losses     = [Losses() for i in range(cfg.REFINE_TIMES)]
        self.distillation_loss = Losses()

        self.inner_iter = 0 
        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def set_inner_iter(self, inner_iter):
        self.inner_iter = inner_iter 

    def forward(self, data, rois, labels):
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois.cpu().numpy() * cfg.DEDUP_BOXES).dot(v)
            
            _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)

            rois  = rois[index, :]


        with torch.set_grad_enabled(self.training):

            backbone_feat      = self.backbone(data)
            box_feat           = self.box_features(backbone_feat, rois)
            mil_score          = self.mil(box_feat)
            refine_score       = self.refinement_agents(box_feat)
            distillation_score = self.distillation(box_feat)


            im_cls_score = mil_score.sum(dim=0, keepdim=True)


            return_dict = {} 
            if self.training:

                return_dict['losses'] = {}

                # image classification loss
                loss_im_cls = mil_loss(im_cls_score, labels)
                return_dict['losses']['loss_im_cls'] = loss_im_cls


                # refinement loss
                boxes = rois[:, 1:]
                im_labels = labels 

                plot_dict = {}
               
                
                lambda_gt, lambda_ign  = get_adaptative_lambda(self.inner_iter)
                return_dict['delta'] = lambda_gt


                for i_refine, refine in enumerate(refine_score):
                    if i_refine == 0:
                        refinement_output = Refinement(boxes, mil_score, im_labels, refine, lambda_gt=lambda_gt, lambda_ign=lambda_ign)
                    else:
                        refinement_output = Refinement(boxes, refine_score[i_refine - 1],
                                         im_labels, refine, lambda_gt=lambda_gt, lambda_ign=lambda_ign)

                    refine_loss = self.Refine_Losses[i_refine](refine,
                                                               refinement_output['labels'],
                                                               refinement_output['cls_loss_weights'],
                                                               refinement_output['gt_assignment'],
                                                               refinement_output['im_labels_real'])

                    return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()


                distilation_sup = (refine_score[0] + refine_score[1]  + refine_score[2])/3


                distillation_refinement = Refinement(boxes, distilation_sup, im_labels, distillation_score, lambda_gt=lambda_gt, lambda_ign=lambda_ign)

                distillation_loss = self.distillation_loss(distillation_score,
                                               distillation_refinement['labels'],
                                               distillation_refinement['cls_loss_weights'],
                                               distillation_refinement['gt_assignment'],
                                               distillation_refinement['im_labels_real'])

                return_dict['losses']['distillation_loss'] = distillation_loss.clone()

                # pytorch0.4 bug on gathering scalar(0-dim) tensors
                for k, v in return_dict['losses'].items():
                    return_dict['losses'][k] = v.unsqueeze(0)
            else:

                
                final_scores = refine_score[0]
                for i in range(1, cfg.REFINE_TIMES):
                    final_scores += refine_score[i]
                
                final_scores += distillation_score
                final_scores /= cfg.REFINE_TIMES + 1

                if cfg.DEDUP_BOXES > 0:
                    # Map scores and predictions back to the original set of boxes
                    final_scores = final_scores[inv_index, :]

                return_dict['final_scores'] = final_scores

            return return_dict

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

def loot_model(args):
    print("Using model description:", args.model)
    model = DetectionModel()
    return model


