import torch
from   tasks.config import cfg
from torchvision.ops import nms
from pdb import set_trace as pause

def box_results_for_corloc(scores, boxes):  

    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]

    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        max_ind = scores[:, j].argmax()

        cls_boxes[j] = torch.cat((boxes[max_ind, :].reshape(1, -1), scores[max_ind, j].reshape(1, -1)), dim=1)

    return cls_boxes


def box_results_with_nms_and_limit(scores, boxes):

    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]
    
    for j in range(1, num_classes):
        inds = torch.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]

        keep = nms(boxes_j, scores_j, cfg.TEST.NMS)
        
        dets_j = torch.cat((boxes_j, scores_j.reshape(-1, 1)), dim=1)
        nms_dets = dets_j[keep, :]

        cls_boxes[j] = nms_dets

    if cfg.TEST.DETECTIONS_PER_IM > 0:

        data = []
        for j in range(1, num_classes):
            data.append(cls_boxes[j][:, -1])

        image_scores = torch.cat(data)

        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = torch.sort(image_scores)[0][-cfg.TEST.DETECTIONS_PER_IM]
           
            for j in range(1, num_classes):
                keep = torch.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]


    return cls_boxes