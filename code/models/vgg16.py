import os

import torch
import torch.nn as nn

from tasks.config import cfg


class VGG16Backbone(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.MaxPool2d(kernel_size=2, stride=2)
								)

		self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.MaxPool2d(kernel_size=2, stride=2)
								)

		self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.MaxPool2d(kernel_size=2, stride=2)
								)

		self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
								nn.ReLU(inplace=True)
								)
		
		self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
								nn.ReLU(inplace=True),
								nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
								nn.ReLU(inplace=True)
								)

		self.dim_out = 512

		self.spatial_scale = 1. / 8.

		self._init_modules()

	def _init_modules(self):
		assert cfg.VGG.FREEZE_AT in [0, 2, 3, 4, 5]
		for i in range(1, cfg.VGG.FREEZE_AT + 1):
			freeze_params(getattr(self, 'conv%d' % i))

	def train(self, mode=True):
		# Override
		self.training = mode

		for i in range(cfg.VGG.FREEZE_AT + 1, 6):
			getattr(self, 'conv%d' % i).train(mode)

	def forward(self, x):
		atribs = {}
		for i in range(1, 6):
			x = getattr(self, 'conv%d' % i)(x)
		return x


	def load_pretrained_imagenet_weights(self, model):
		"""Load pretrained weights
		"""
		weights_file = os.path.join(cfg.ROOT_DIR, cfg.VGG.IMAGENET_PRETRAINED_WEIGHTS)

		print("Loading weights from: ", weights_file)
		pretrianed_state_dict = self.convert_state_dict(torch.load(weights_file))

		model_state_dict = model.state_dict()

		for k, v in pretrianed_state_dict.items():
			model_state_dict[k].copy_(v)


	def convert_state_dict(self, src_dict):
		"""Return the correct mapping of tensor name and value

		Mapping from the names of torchvision model to our vgg conv_body and box_head.
		"""
		dst_dict = {}
		dst_dict['backbone.conv1.0.weight'] = src_dict['features.0.weight']
		dst_dict['backbone.conv1.0.bias']   = src_dict['features.0.bias']
		dst_dict['backbone.conv1.2.weight'] = src_dict['features.2.weight']
		dst_dict['backbone.conv1.2.bias']   = src_dict['features.2.bias']
		dst_dict['backbone.conv2.0.weight'] = src_dict['features.5.weight']
		dst_dict['backbone.conv2.0.bias']   = src_dict['features.5.bias']
		dst_dict['backbone.conv2.2.weight'] = src_dict['features.7.weight']
		dst_dict['backbone.conv2.2.bias']   = src_dict['features.7.bias']
		dst_dict['backbone.conv3.0.weight'] = src_dict['features.10.weight']
		dst_dict['backbone.conv3.0.bias']   = src_dict['features.10.bias']
		dst_dict['backbone.conv3.2.weight'] = src_dict['features.12.weight']
		dst_dict['backbone.conv3.2.bias']   = src_dict['features.12.bias']
		dst_dict['backbone.conv3.4.weight'] = src_dict['features.14.weight']
		dst_dict['backbone.conv3.4.bias']   = src_dict['features.14.bias']
		dst_dict['backbone.conv4.0.weight'] = src_dict['features.17.weight']
		dst_dict['backbone.conv4.0.bias']   = src_dict['features.17.bias']
		dst_dict['backbone.conv4.2.weight'] = src_dict['features.19.weight']
		dst_dict['backbone.conv4.2.bias']   = src_dict['features.19.bias']
		dst_dict['backbone.conv4.4.weight'] = src_dict['features.21.weight']
		dst_dict['backbone.conv4.4.bias']   = src_dict['features.21.bias']
		dst_dict['backbone.conv5.0.weight'] = src_dict['features.24.weight']
		dst_dict['backbone.conv5.0.bias']   = src_dict['features.24.bias']
		dst_dict['backbone.conv5.2.weight'] = src_dict['features.26.weight']
		dst_dict['backbone.conv5.2.bias']   = src_dict['features.26.bias']
		dst_dict['backbone.conv5.4.weight'] = src_dict['features.28.weight']
		dst_dict['backbone.conv5.4.bias']   = src_dict['features.28.bias']
		
		dst_dict['box_features.fc1.weight']     = src_dict['classifier.0.weight']
		dst_dict['box_features.fc1.bias']       = src_dict['classifier.0.bias']
		dst_dict['box_features.fc2.weight']     = src_dict['classifier.3.weight']
		dst_dict['box_features.fc2.bias']       = src_dict['classifier.3.bias']

		return dst_dict


def freeze_params(m):
	"""Freeze all the weights by setting requires_grad to False
	"""
	for p in m.parameters():
			p.requires_grad = False
