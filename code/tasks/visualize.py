##  This code is a huge mess, sorry. :)

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle

import torch

import _init_paths 
from tasks.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from tasks.test import empty_results, extend_results
from tasks.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation

from datasets.voc_dataset_evaluator import voc_info
from torchvision.ops.boxes import box_iou

import utils.logging

from pdb import set_trace as pause


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize results =)')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--detections',
        help='the path for result file.')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results.', default=None)

    return parser.parse_args()


if __name__ == '__main__':

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert os.path.exists(args.detections)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    if args.dataset == 'voc2007test':
        cfg.TEST.DATASETS = ('voc_2007_test',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012test':
        cfg.TEST.DATASETS = ('voc_2012_test',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2007trainval':
        cfg.TEST.DATASETS = ('voc_2007_trainval',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012trainval':
        cfg.TEST.DATASETS = ('voc_2012_trainval',)
        cfg.MODEL.NUM_CLASSES = 20
    else: 
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    logger.info('Re-evaluation with config:')
    logger.info(pprint.pformat(cfg))

    with open(args.detections, 'rb') as f:
        results = pickle.load(f)
        logger.info('Loading results from {}.'.format(args.detections))
    all_boxes = results['all_boxes']

    dataset_name = cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    final_boxes = empty_results(num_classes, num_images)
    test_corloc = 'train' in dataset_name


    info_ann = voc_info(dataset)
    ann_cachedir = os.path.join(info_ann['devkit_path'], 'annotations_cache_{}'.format(info_ann['year']))
    annotations_cache_file = os.path.join(ann_cachedir, 'test_annots.pkl')
    
    # read list of images
    with open(annotations_cache_file, 'rb') as f:
        annotations  = pickle.load(f)
    
    img_keys = []
    for i, entry in enumerate(roidb):

        boxes = all_boxes[entry['image']]
        if test_corloc and not save_detections:
            _, _, cls_boxes_i = box_results_for_corloc(boxes['scores'], boxes['boxes'])
        else:
            _, _, cls_boxes_i = box_results_with_nms_and_limit(boxes['scores'],  boxes['boxes'])
        extend_results(i, final_boxes, cls_boxes_i)
        img_keys.append(entry['image'])
    

    output = {}

    for c in range(len(final_boxes)):

        for img_idx in range(len(final_boxes[c])):

            img_key = img_keys[img_idx].split('/')[-1].split('.')[0]

            
            boxes   = final_boxes[c][img_idx]

            if img_key not in output:
                output[img_key] = {}

            output[img_key][classes[c]] = boxes



    for img_key in img_keys:

        short_key =  img_key.split("/")[-1].split('.')[0]

        img = cv2.imread(img_key)

        img_detections = output[short_key]

        img_annotations = annotations[short_key]
        

        gt_boxes = []
        gt_classes = []
        for annotation in img_annotations:
            
            box = annotation['bbox']
            gt_boxes.append(box)
            gt_classes.append(annotation['name'])

        det_boxes = []
        det_classes = []
        det_scores = []

        for det_cls in img_detections:

            boxes = img_detections[det_cls]

            if len(boxes) == 0:
                continue

            for b in range(boxes.shape[0]):

                box = boxes[b,:4]
                score = boxes[b,4]

                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[2]), int(box[3]))

                if score > 0.3:
                    det_boxes.append(box)
                    det_classes.append(det_cls)
                    det_scores.append(score)
        
        if len(det_boxes) == 0:
            continue
        gt_boxes = torch.tensor(gt_boxes).float()
        det_boxes = torch.tensor(det_boxes).float()

        overlaps = box_iou(gt_boxes, det_boxes)
        

        used_detections = []
        iou, det_box_idxs = overlaps.max(dim=1)

        iou_sorted, gt_box_idxs = iou.sort(descending=True)


        aft_det_boxes     = []
        aft_det_boxes_txt = []
        aft_det_color     = []


        for idx, gt_i in enumerate(gt_box_idxs):

            gt_box = gt_boxes[gt_i].tolist()

            best_det_idx = det_box_idxs[gt_i]

            det_box = det_boxes[best_det_idx]
            det_iou = iou_sorted[idx]
            det_class = det_classes[best_det_idx]
            det_score = det_scores[best_det_idx]

            if best_det_idx not in used_detections and det_iou > 0:
                # in this case this detection box was never used
                used_detections.append(best_det_idx)

                gt_p1 = (int(gt_box[0]), int(gt_box[1]))
                gt_p2 = (int(gt_box[2]), int(gt_box[3]))

              
                det_color = (0,255,0)
                
                if det_iou < 0.5:
                    det_color = (0,0,255)

                img = cv2.rectangle(img, gt_p1, gt_p2, (255,0,0), 4)

                text = "{:s} {:2d}%".format(det_class, int(det_score*100))

                aft_det_boxes.append(det_box)
                aft_det_boxes_txt.append(text)
                aft_det_color.append(det_color)

            else:
                # if we fall here, gt box dont have a detection box
                gt_p1 = (int(gt_box[0]), int(gt_box[1]))
                gt_p2 = (int(gt_box[2]), int(gt_box[3]))

                img = cv2.rectangle(img, gt_p1, gt_p2, (0,255,255), 4)


        for b_im, det_box in enumerate(aft_det_boxes):

            dt_p1 = (int(det_box[0]), int(det_box[1]))
            dt_p2 = (int(det_box[2]), int(det_box[3]))

            text = aft_det_boxes_txt[b_im]
            det_color = aft_det_color[b_im]

            img = cv2.rectangle(img, dt_p1, dt_p2, det_color, 4)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 1
            
            (text_width, text_height)  = cv2.getTextSize(text, font, font_scale, thickness)[0]


            text_offset_x = dt_p1[0]
            text_offset_y = dt_p1[1]+text_height


            # make the coords of the box with a small padding of two pixels
            box_coords = ((text_offset_x, text_offset_y+2), (text_offset_x + text_width + 2, text_offset_y - text_height))
            
            cv2.rectangle(img, box_coords[0], box_coords[1], det_color, cv2.FILLED)

            cv2.putText(img, text, (text_offset_x,text_offset_y) , font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

        if args.output_dir is not None: 
            save_at = img_key.replace('data/VOCdevkit/VOC2007/JPEGImages', args.output_dir).replace('.jpg','.png')
            print(save_at)
            cv2.imwrite(save_at, img)
        else:
            cv2.imshow("yey", img)
            if cv2.waitKey(0) == ord('q'):
                exit()
