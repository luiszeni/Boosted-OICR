# inspired in : https://raw.githubusercontent.com/pytorch/vision/master/torchvision/datasets/voc.py
import os
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset
import xml.etree.ElementTree as ET
import cv2
from pdb import set_trace as pause
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torch
from torch.utils.data import Dataset
from six.moves import cPickle as pickle
import numpy as np
import utils.boxes as box_utils
from tasks.config    import cfg
import logging
logger = logging.getLogger(__name__)


DATASET_YEAR_DICT = {
	'2012': {
		'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
		'filename': 'VOCtrainval_11-May-2012.tar',
		'md5': '6cd6e144f989b92b3379bac3b3de84fd',
		'base_dir': os.path.join('VOCdevkit', 'VOC2012')
	},
	'2007': {
		'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
		'filename': 'VOCtrainval_06-Nov-2007.tar',
		'md5': 'c52e279531787c972589f7e41ab4ae64',
		'base_dir': os.path.join('VOCdevkit', 'VOC2007')
	},
	'2007-test': {
		'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
		'filename': 'VOCtest_06-Nov-2007.tar',
		'md5': 'b6e924de25625d8de591ea690078ad9f',
		'base_dir': os.path.join('VOCdevkit', 'VOC2007')
	}
}

class VOCDetection(Dataset):
	def __init__(self,
				 root,
				 year='2007',
				 image_set='trainval',
				 download=False,
				 transforms=None):

		self.root = root

		self.transforms = transforms
		self.year = year
		if year == "2007" and image_set == "test":
			year = "2007-test"
		self.url = DATASET_YEAR_DICT[year]['url']
		self.filename = DATASET_YEAR_DICT[year]['filename']
		self.md5 = DATASET_YEAR_DICT[year]['md5']
		valid_sets = ["train", "trainval", "val"]
		if year == "2007-test":
			valid_sets.append("test")
		self.image_set = verify_str_arg(image_set, "image_set", valid_sets)

		base_dir = DATASET_YEAR_DICT[year]['base_dir']
		voc_root = os.path.join(self.root, base_dir)
		image_dir = os.path.join(voc_root, 'JPEGImages')
		annotation_dir = os.path.join(voc_root, 'Annotations')

		if download:
			download_extract(self.url, self.root, self.filename, self.md5)

		# pause()
		if not os.path.isdir(voc_root):
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')

		splits_dir = os.path.join(voc_root, 'ImageSets/Main')

		split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

		with open(os.path.join(split_f), "r") as f:
			file_names = [x.strip() for x in f.readlines()]



		self.class_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

		self.num_classes  = len(self.class_labels)

		self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
		annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
		assert (len(self.images) == len(annotations))


		#load annotations:
		self.annotations = []
		for i, ann_path in enumerate(annotations):
			raw_annot = self.parse_voc_xml(ET.parse(ann_path).getroot())['annotation']
			self.annotations.append({
				'img_key': int(raw_annot['filename'].replace('.jpg','')),
				'width'  : int(raw_annot['size']['width']),
				'height' : int(raw_annot['size']['height']),
				'object'  : raw_annot['object']
				})
		


		##TODO ajust this to be more beauty =p
		ss_data = self.root + 'selective_search_data/voc_' + self.year + '_' + self.image_set + '.pkl'
		with open(ss_data, 'rb') as f:
			proposals = pickle.load(f)

		sort_proposals(proposals, 'indexes')

		self.proposals = []
		for i, boxes in enumerate(proposals['boxes']):
			if i % 2500 == 0:
				logger.info(' {:d}/{:d}'.format(i + 1, len(proposals['boxes'])))
			
			annotation = self.annotations[i]
			assert annotation['img_key'] == proposals['indexes'][i]
			# Remove duplicate boxes and very small boxes and then take top k
			boxes = box_utils.clip_boxes_to_image(boxes, annotation['height'], annotation['width'])
			keep = box_utils.unique_boxes(boxes)
			boxes = boxes[keep, :]
			keep = box_utils.filter_small_boxes(boxes, cfg.FAST_RCNN.MIN_PROPOSAL_SIZE)
			boxes = boxes[keep, :]
			if cfg.FAST_RCNN.TOP_K > 0:
				boxes = boxes[:cfg.FAST_RCNN.TOP_K, :]
			self.proposals.append(boxes.astype(np.float))



	def __getitem__(self, index):
		
		img_idx, img_scale, hflip, batch_idx = index
		# print(img_scale, hflip)

		img_key   = self.images[img_idx]
		img       = cv2.imread(img_key)
		
		target    = self.annotations[img_idx]['object']
		proposals = self.proposals[img_idx]


		to_transform = {'img':img,
						'proposals': proposals,
						'img_scale': img_scale,
						'hflip': hflip
						}

		if self.transforms is not None:
			to_transform = self.transforms(to_transform)

		img       = to_transform['img']
		proposals = to_transform['proposals']

		return img, self.get_class_labels(target), proposals, img_key,  self.proposals[img_idx]

	def __len__(self):
		return len(self.images)


	def get_class_labels(self, target):

		classification_labels = torch.zeros(self.num_classes)
		
		for ann in target:
			classification_labels[self.class_labels.index(ann['name'])] = 1

		return classification_labels


	def parse_voc_xml(self, node):
		voc_dict = {}
		children = list(node)
		if children:
			def_dic = collections.defaultdict(list)
			for dc in map(self.parse_voc_xml, children):
				for ind, v in dc.items():
					def_dic[ind].append(v)
			if node.tag == 'annotation':
				def_dic['object'] = [def_dic['object']]
			voc_dict = {
				node.tag:
					{ind: v[0] if len(v) == 1 else v
					 for ind, v in def_dic.items()}
			}
		if node.text:
			text = node.text.strip()
			if not children:
				voc_dict[node.tag] = text
		return voc_dict

def sort_proposals(proposals, id_field):
	"""Sort proposals by the specified id field."""
	order = np.argsort(proposals[id_field])
	fields_to_sort = ['boxes', id_field, 'scores']
	for k in fields_to_sort:
		proposals[k] = [proposals[k][i] for i in order]

def download_extract(url, root, filename, md5):
	download_url(url, root, filename, md5)
	with tarfile.open(os.path.join(root, filename), "r") as tar:
		tar.extractall(path=root)