import numpy as np
import torch
from   tasks.config import cfg
from torchvision.ops import nms
from pdb import set_trace as pause

def box_results_for_corloc(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results for CorLoc evaluation.

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        max_ind = np.argmax(scores[:, j])
        cls_boxes[j] = np.hstack((boxes[max_ind, :].reshape(1, -1),
                               np.array([[scores[max_ind, j]]])))

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def box_results_with_nms_and_limit(scores, boxes):

    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]
    
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        keep = nms(torch.tensor(dets_j), torch.tensor(scores_j), torch.tensor(cfg.TEST.NMS))
        nms_dets = dets_j[keep.numpy(), :]

        cls_boxes[j] = nms_dets

    if cfg.TEST.DETECTIONS_PER_IM > 0:

        data = []
        for j in range(1, num_classes):
            data.append(cls_boxes[j][:, -1])

        image_scores = np.hstack(data)

        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
           
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes      = im_results[:, :-1]
    scores     = im_results[:, -1]

    return scores, boxes, cls_boxes

