import init_paths
import argparse
import time

import os
import yaml
import pprint
import torch
import sys
from tqdm import tqdm



from models import *
from datasets.pascal_voc import VOCDetection
import datasets.voc_dataset_evaluator as voc_dataset_evaluator

from tasks.config    import cfg, load_config

import utils.data_transforms as dt_trans

from utils.data_samplers           import TestSampler, collate_minibatch
from utils.detectron_weight_helper import load_detectron_weight
from utils.training_stats import TrainingStats

from utils.test_utils import box_results_for_corloc, box_results_with_nms_and_limit

import logging
from utils.logging import setup_logging
logger = logging.getLogger(__name__)

from pdb import set_trace as pause
from utils.misc import *

def get_all_detections(args, dataset, early_stop=False):


	det_name = 'detections.pkl'
	if 'train' in dataset.image_set:
			det_name = 'discovery.pkl'

	det_file = os.path.join(args.output_dir, det_name)

	if os.path.exists(det_file):
		print('det_file', det_file, 'exits. I will use it')
		return load_object(det_file)['all_boxes']

	print('Creating detections...')
	model = initialize_model_from_cfg(args)
	num_images = len(dataset)
	num_classes = cfg.MODEL.NUM_CLASSES
	detections = {}

	batchSampler = TestSampler(num_images)

	if early_stop:
		batchSampler = TestSampler(11)
	
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_sampler = batchSampler,
		num_workers   = cfg.DATA_LOADER.NUM_THREADS,
		collate_fn    = collate_minibatch)

	for data in tqdm(dataloader):		
		image, target, box_proposals, img_key, original_proposals = data
		
		img_key = img_key[0]
		
		if len(box_proposals) == 0:
			continue
	  
		return_dict = model(image.cuda(), box_proposals.cuda(), target.cuda())

		scores = return_dict['final_scores'].cpu()

		if img_key not in detections:
			detections[img_key] = scores[None]
		else:
			detections[img_key] = torch.cat((detections[img_key], scores[None]))

	for img_key in detections:
		detections[img_key] = detections[img_key].mean(dim=0)
	
		
	cfg_yaml = yaml.dump(cfg)


	save_object(
		dict(
			all_boxes=detections,
			cfg=cfg_yaml
		), det_file
	)
	logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
	
	return detections


def test_net_on_dataset(args, dataset, detections, use_matlab = True, early_stop=False):	
	
	print('Running test on detections...')
	num_images = len(dataset)
	num_classes = dataset.num_classes + 1
	final_boxes = empty_results(num_classes, num_images)
	test_corloc = 'train' in dataset.image_set

	for i in tqdm(range(num_images)):

		if early_stop and i > 10: break

		detect = detections[dataset.images[i]]
		proposals = torch.tensor(dataset.proposals[i], dtype=torch.float32)

		if detect is not None:
			if test_corloc:
				cls_boxes_i = box_results_for_corloc(detect, proposals)
			else:
				cls_boxes_i = box_results_with_nms_and_limit(detect, proposals)

			extend_results(i, final_boxes, cls_boxes_i)
		else:
			final_boxes = None

	del detections
	del cls_boxes_i

	voc_eval = voc_dataset_evaluator.evaluate_boxes(dataset, final_boxes, args.output_dir, test_corloc=test_corloc, use_matlab=use_matlab)

	return voc_eval



def initialize_model_from_cfg(args):
	model = eval(args.model).loot_model(args)
	model.eval()

	model.cuda()

	if args.load_ckpt:
		load_name = args.load_ckpt
		logger.info("loading checkpoint %s", load_name)
		checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
		
		model.load_state_dict(checkpoint['model'], strict=True)

	return model


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
	parser = argparse.ArgumentParser(description='Test Detection network')
	parser.add_argument('--dataset', help='training dataset')
	parser.add_argument('--cfg', dest='cfg_file', required=True, help='optional config file')

	parser.add_argument('--load_ckpt', help='path of checkpoint to load')

	parser.add_argument('--output_dir', help='output directory to save the testing results. If not provided, defaults to [args.load_ckpt|args.load_detectron]/../test.')

	parser.add_argument('--set', dest='set_cfgs', help='set config keys, will overwrite config in the cfg_file. See lib/core/config.py for all options', default=[], nargs='*')

	parser.add_argument('--model', help='Set model', type=str)

	parser.add_argument('--early_stop', help='run eval only to 10 images', action='store_true')
	
	return parser.parse_args()


if __name__ == '__main__':

	if not torch.cuda.is_available():
		sys.exit("Need a CUDA device to run the code.")

	logger = setup_logging(__name__)
	args = parse_args()
	logger.info('Called with args:')
	logger.info(args)

	if args.output_dir is None:
		ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
		args.output_dir = os.path.join(
			os.path.dirname(os.path.dirname(ckpt_path)), 'test',
			os.path.basename(ckpt_path).split('.')[0])
		logger.info('Automatically set output directory to %s', args.output_dir)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)


	load_config(args.cfg_file)

	if args.dataset == 'voc2007test':
		dataset_name   = 'voc'
		dataset_year   = '2007'
		dataset_subset = 'test'

	elif args.dataset == 'voc2012test':
		dataset_name   = 'voc'
		dataset_year   = '2012'
		dataset_subset = 'test'

	elif args.dataset == 'voc2007trainval':
		dataset_name   = 'voc'
		dataset_year   = '2007'
		dataset_subset = 'trainval'

	elif args.dataset == 'voc2012trainval':
		dataset_name   = 'voc'
		dataset_year   = '2012'
		dataset_subset = 'trainval'
	else:  # UOPS
		assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'

	logger.info('Testing with config:')
	logger.info(pprint.pformat(cfg))

	if args.load_ckpt:
		while not os.path.exists(args.load_ckpt):
			logger.info('Waiting for {} to exist...'.format(args.load_ckpt))
			time.sleep(10)



	transforms = dt_trans.Compose([
		dt_trans.Normalize([102.9801, 115.9465, 122.7717]),
		dt_trans.HorizontalFlip(),
		dt_trans.Resize(),
		dt_trans.ToTensor(),
		
		])

	if dataset_name == 'voc':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year=dataset_year, image_set=dataset_subset, transforms=transforms) 

	detections = get_all_detections(args, dataset, early_stop= args.early_stop)

	test_net_on_dataset(args, dataset, detections, use_matlab=True, early_stop= args.early_stop)
