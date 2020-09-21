"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import yaml
import time
import datetime

import torch

import numpy as np

import _init_paths

import utils.logging

from tasks.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg

from utils.timer import Timer

from models import *

from utils.io    import save_object, load_object
from utils.timer import Timer

import utils.boxes as box_utils
import utils.blob  as blob_utils
import utils.image as image_utils
import utils.net   as net_utils

from datasets              import task_evaluation
from datasets.json_dataset import JsonDataset

from collections import defaultdict


from torchvision.ops import nms


from pdb import set_trace as pause

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def im_detect_all(model, im, box_proposals=None, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    scores, boxes, im_scale  = im_detect_bbox_aug(model, im, box_proposals)
    timers['im_detect_bbox'].toc()

    return {'scores':scores, 'boxes':boxes}


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    

    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        
        _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)

        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    data = torch.from_numpy(inputs['data']).cuda()
    rois = torch.from_numpy(inputs['rois']).cuda()
    labels = torch.from_numpy(inputs['labels']).cuda()

    return_dict = model(data,rois,labels)


    if 'final_scores' in return_dict:
        # print("pickle rick")
        

        scores = return_dict['final_scores'].cpu().numpy().squeeze()
        scores = scores.reshape([-1, scores.shape[-1]])
        pred_boxes = boxes

        if cfg.DEDUP_BOXES > 0:
            # Map scores and predictions back to the original set of boxes
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

    else:
        scores = None
        pred_boxes = None
        im_scale = None

    return scores, pred_boxes, im_scale


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, 'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', 'Score heuristic must be union whenever coord heuristic is union'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts  = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals=box_proposals)
        
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(model, im, scale, max_size, box_proposals)
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(model, im, scale, max_size, box_proposals, hflip=True)
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(model, im, aspect_ratio, box_proposals)
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(model, im, aspect_ratio, box_proposals, hflip=True)
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals)

    add_preds_t(scores_i, boxes_i)


    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        if scores_ts[0] is not None:
            scores_c = np.mean(scores_ts, axis=0)
        else:
            scores_c = None

    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i


def im_detect_bbox_hflip(
        model, im, target_scale, target_max_size, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)

    scores_hf, boxes_hf, im_scale = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf
    )

    if boxes_hf is not None:
        # Invert the detections computed on the flipped image
        boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)
    else:
        boxes_inv = None

    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
        model, im, target_scale, target_max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _, = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


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

        keep = nms(torch.tensor(boxes_j), torch.tensor(scores_j), cfg.TEST.NMS).numpy()
        nms_dets = dets_j[keep, :]

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


def _get_rois_blob(im_rois, im_scale):
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _get_blobs(im, rois, target_scale, target_max_size):
    blobs = {}
    blobs['data'], im_scale = blob_utils.get_image_blob(im, target_scale, target_max_size)
    
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    
    blobs['labels'] = np.zeros((1, cfg.MODEL.NUM_CLASSES), dtype=np.int32)
    return blobs, im_scale


def get_eval_functions():
    # Determine which parent or child function should handle inference
    # Generic case that handles all network types other than RPN-only nets
    # and RetinaNet
    child_func  = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        if not (dataset_name + '.pkl') in cfg.TEST.PROPOSAL_FILES[index]:
            proposal_file = os.path.join(cfg.TEST.PROPOSAL_FILES[index],
                                         dataset_name + '.pkl')
        else:
            proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False, use_matlab=False, early_stop=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            print("is parent")
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                
                results = parent_func(args, dataset_name, proposal_file, output_dir, multi_gpu=multi_gpu_testing, use_matlab=use_matlab, early_stop=early_stop)

                all_results.update(results)

            return all_results
        else:
            print("is child")
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(args, dataset_name, proposal_file, output_dir, multi_gpu=False, gpu_id=0, use_matlab = False, early_stop=False):

    
    # print("test_net_on_dataset")    
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    
    test_timer.tic()
    
    all_boxes = test_net(args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id, early_stop=early_stop)
    test_timer.toc()

    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))

    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    final_boxes = empty_results(num_classes, num_images)
    test_corloc = 'train' in dataset_name
   

    all_cls_scores = {}

    for i, entry in enumerate(roidb):

        if early_stop and i > 10: break

        boxes = all_boxes[entry['image']]
        
        cls_key = entry['image'].replace('.jpg','').split('/')[-1]

        # print(cls_key)

        if boxes['scores'] is not None:
            if test_corloc:
                # print("corlooking")
                _, _, cls_boxes_i = box_results_for_corloc(boxes['scores'], boxes['boxes'])
            else:
                _, _, cls_boxes_i = box_results_with_nms_and_limit(boxes['scores'], boxes['boxes'])

            extend_results(i, final_boxes, cls_boxes_i)
        else:
            final_boxes = None
    
    results = task_evaluation.evaluate_all(dataset, final_boxes, output_dir, test_corloc, use_matlab = use_matlab)
    return results



def test_net(args, dataset_name, proposal_file, output_dir, ind_range=None, gpu_id=0, early_stop=False):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    # print('test_net')
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(dataset_name, proposal_file, ind_range)
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes = {}

    timers = defaultdict(Timer)
    
    


    if 'train' in dataset_name:
        if ind_range is not None:
            det_name = 'discovery_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'discovery.pkl'
    else:
        if ind_range is not None:
            det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'detections.pkl'
    
    det_file = os.path.join(output_dir, det_name)
    if os.path.exists(det_file):
        print('the file', det_file, 'exists. I am loading detections from it...')
        return load_object(det_file)['all_boxes']

    for i, entry in enumerate(roidb):
        if early_stop and i > 10: break

        box_proposals = entry['boxes']
        if len(box_proposals) == 0:
            continue
       
        im = cv2.imread(entry['image'])
        # print(entry['image'])
        cls_boxes_i = im_detect_all(model, im, box_proposals, timers)

        all_boxes[entry['image']] = cls_boxes_i

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            det_time = (timers['im_detect_bbox'].average_time)
            
            logger.info(('im_detect: range [{:d}, {:d}] of {:d}:{:d}/{:d} {:.3f}s (eta: {})').format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1, start_ind + num_images, det_time, eta))

    cfg_yaml = yaml.dump(cfg)

    save_object(
        dict(
            all_boxes=all_boxes,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = eval(args.model).loot_model(args)
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)


    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', help='training dataset')
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='optional config file')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument('--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument('--output_dir', help='output directory to save the testing results. If not provided, defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument('--set', dest='set_cfgs', help='set config keys, will overwrite config in the cfg_file. See lib/core/config.py for all options', default=[], nargs='*')
    parser.add_argument('--range', help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
    parser.add_argument('--multi-gpu-testing', help='using multiple gpus for inference', action='store_true')
    parser.add_argument('--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument('--model', help='Set model', type=str)

    parser.add_argument('--use_matlab', help='use matlab?', action='store_true')
    parser.add_argument('--early_stop', help='run eval only to 10 images', action='store_true')
    



    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test',
            os.path.basename(ckpt_path).split('.')[0])
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

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
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    if args.load_ckpt:
        while not os.path.exists(args.load_ckpt):
            logger.info('Waiting for {} to exist...'.format(args.load_ckpt))
            time.sleep(10)
    if args.load_detectron:
        while not os.path.exists(args.load_detectron):
            logger.info('Waiting for {} to exist...'.format(args.load_detectron))
            time.sleep(10)

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True,
        use_matlab=args.use_matlab,
        early_stop=args.early_stop)
