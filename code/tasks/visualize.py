import init_paths
import argparse
import time
import cv2

import os
import yaml
import pprint
import torch
import sys
from tqdm import tqdm
from torchvision.ops.boxes import box_iou
import numpy as np

from models import *
from datasets.pascal_voc import VOCDetection

import datasets.voc_dataset_evaluator as voc_dataset_evaluator

from tasks.config    import cfg, load_config

import utils.data_transforms as dt_trans

from utils.data_samplers           import VisualizeSampler, collate_minibatch_visualize
from utils.detectron_weight_helper import load_detectron_weight
from utils.training_stats import TrainingStats

from utils.test_utils import box_results_for_corloc, box_results_with_nms_and_limit

import logging
from utils.logging import setup_logging
logger = logging.getLogger(__name__)

from pdb import set_trace as pause
from util.misc import *


def gibe_me_cool_visualiztions(args, dataset):

	num_classes = cfg.MODEL.NUM_CLASSES
	

	detections_file = os.path.abspath(args.detections)
	detections = load_object(detections_file)

	batchSampler = VisualizeSampler(len(dataset))

	
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_sampler = batchSampler,
		num_workers   = cfg.DATA_LOADER.NUM_THREADS,
		collate_fn    = collate_minibatch_visualize)

	for i, data in enumerate(dataloader):		
		image, target, box_proposals, img_key, original_proposals = data
		img_orig= image[0].copy()
		img     = image[0]
		img_key = img_key[0]

		proposals = dataset.proposals[i]
		detect = detections[img_key]


		_, _, cls_boxes_i = box_results_with_nms_and_limit(detect.numpy(), proposals)


		img_detections = cls_boxes_i

		img_annotations = dataset.annotations[i]
			

		gt_boxes = []
		gt_classes = []
		for annotation in img_annotations['object']:
			box = annotation['bndbox']
			gt_boxes.append([ int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax']) ]) 
			gt_classes.append(annotation['name'])


		det_boxes = []
		det_classes = []
		det_scores = []


		for c, det_cls in enumerate(img_detections):

			if len(det_cls) == 0:
				continue

			for b in range(det_cls.shape[0]):

				box = det_cls[b,:4]
				score = det_cls[b,4]

				p1 = (int(box[0]), int(box[1]))
				p2 = (int(box[2]), int(box[3]))

				if score > 0.3:
					det_boxes.append(box)
					det_classes.append(c-1)
					det_scores.append(score)
		
		if len(det_boxes) == 0:
			continue


		gt_boxes  = torch.tensor(gt_boxes).float()
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
				used_detections.append(best_det_idx)

				gt_p1 = (int(gt_box[0]), int(gt_box[1]))
				gt_p2 = (int(gt_box[2]), int(gt_box[3]))

				det_color = (0,255,0)
				
				if det_iou < 0.5:
					det_color = (0,0,255)

				img = cv2.rectangle(img, gt_p1, gt_p2, (255,0,0), 4)

				text = "{:s} {:2d}%".format(dataset.class_labels[det_class], int(det_score*100))

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

		if args.output_dir is  None: 
			cv2.imshow("Cool detections", img)
			if cv2.waitKey(0) == ord('q'):
				exit()
		else:
			save_at = os.path.join(args.output_dir, img_key.split('/')[-1].replace('.jpg','.png'))
			print("saving:", save_at)
			cv2.imwrite(save_at, img)
				
	return detections



def parse_args():
	"""Parse in command line arguments"""
	parser = argparse.ArgumentParser(description='Visualize detections')
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
		help='output directory to save the testing results.')
	parser.add_argument(
		   '--set', dest='set_cfgs',
		   help='set config keys, will overwrite config in the cfg_file.'
				' See lib/core/config.py for all options',
		   default=[], nargs='*')
	
	return parser.parse_args()


if __name__ == '__main__':

	if not torch.cuda.is_available():
		sys.exit("Need a CUDA device to run the code.")

	logger = setup_logging(__name__)
	args = parse_args()

	load_config(args.cfg_file)

	logger.info('Visualizing with config:')
	logger.info(pprint.pformat(cfg))



	if args.dataset == 'voc2007test':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year='2007', image_set='test')
	elif args.dataset == 'voc2012test':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year='2012', image_set='test')
	elif args.dataset == 'voc2007trainval':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year='2007', image_set='trainval')
	elif args.dataset == 'voc2012trainval':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year='2012', image_set='trainval')
	else:  # UOPS
		assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'


	detections = gibe_me_cool_visualiztions(args, dataset)