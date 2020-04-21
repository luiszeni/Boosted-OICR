import argparse
import cv2
import os
import pprint
from six.moves import cPickle as pickle
import xml.etree.ElementTree as ET
import torch

import _init_paths

from tasks.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from tasks.test import empty_results, extend_results
from tasks.test import box_results_with_nms_and_limit
from datasets.json_dataset import JsonDataset

from datasets.voc_dataset_evaluator import voc_info
from torchvision.ops.boxes import box_iou

import utils.logging

from pdb import set_trace as pause


def parse_args():
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


def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


if __name__ == '__main__':

	logger = utils.logging.setup_logging(__name__)
	args = parse_args()
	logger.info('Called with args:')
	logger.info(args)



	assert os.path.exists(args.detections)

	save_detections = True

	if args.output_dir is None:
		save_detections = False
		logger.info('No path loation informed, I will display the detections in opencv window')

	if args.cfg_file is not None:
		merge_cfg_from_file(args.cfg_file)

	if args.set_cfgs is not None:
		merge_cfg_from_list(args.set_cfgs)

	classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

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

	logger.info('Visualizing with following configs:')
	logger.info(pprint.pformat(cfg))


	# load detections
	logger.info('Loading detections from:' + args.detections)
	with open(args.detections, 'rb') as f:
		results = pickle.load(f)
	
	all_boxes = results['all_boxes']


	dataset = JsonDataset(cfg.TEST.DATASETS[0])
	roidb = dataset.get_roidb()
	num_images = len(roidb)
	num_classes = cfg.MODEL.NUM_CLASSES + 1
	final_boxes = empty_results(num_classes, num_images)

	#load GT annotations

	info_ann = voc_info(dataset)
	ann_cachedir = os.path.join(info_ann['devkit_path'], 'annotations_cache_{}'.format(info_ann['year']))

	if not os.path.isdir(ann_cachedir):
		os.mkdir(ann_cachedir)

	imagesetfile = os.path.join(info_ann['devkit_path'], 'VOC' + info_ann['year'], 'ImageSets', 'Main', info_ann['image_set'] + '.txt')

	imageset  = os.path.splitext(os.path.basename(imagesetfile))[0]
	annotations_cache_file = os.path.join(ann_cachedir, info_ann['image_set'] + '_annots.pkl')
	# read list of images
	with open(imagesetfile, 'r') as f:
		lines = f.readlines()
	imagenames = [x.strip() for x in lines]

	if not os.path.isfile(annotations_cache_file):
		# load annots
		annotations = {}
		for i, imagename in enumerate(imagenames):
			annotations[imagename] = parse_rec(info_ann['anno_path'].format(imagename))
			if i % 100 == 0:
				logger.info(
					'Reading annotation for {:d}/{:d}'.format(
						i + 1, len(imagenames)))
		# save
		logger.info('Saving cached annotations to {:s}'.format(annotations_cache_file))
		with open(annotations_cache_file, 'wb') as f:
			pickle.dump(annotations, f, pickle.HIGHEST_PROTOCOL)
	else:
		# load
		with open(annotations_cache_file, 'rb') as f:
			annotations = pickle.load(f)


	# apply nms into the detected boxes.

	img_keys = []
	for i, entry in enumerate(roidb):
		boxes = all_boxes[entry['image']]
		_, _, cls_boxes_i = box_results_with_nms_and_limit(boxes['scores'],  boxes['boxes'], boxes['cls_scores'])
		extend_results(i, final_boxes, cls_boxes_i)
		img_keys.append(entry['image'])


	

	# convert keys.
	output = {}

	for c in range(len(final_boxes)):

		for img_idx in range(len(final_boxes[c])):

			img_key = img_keys[img_idx].split('/')[-1].split('.')[0]

			
			boxes   = final_boxes[c][img_idx]

			if img_key not in output:
				output[img_key] = {}

			output[img_key][classes[c]] = boxes


	# create the visualization (I know it is a mess...)
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

	

		if save_detections: 
			save_at = os.path.join(args.output_dir, img_key.split('/')[-1].replace('.jpg','.png'))
			print("saving:", save_at)
			cv2.imwrite(save_at, img)
		else:
			cv2.imshow("cursed sanity", img)
			if cv2.waitKey(0) == ord('q'):
				exit()

