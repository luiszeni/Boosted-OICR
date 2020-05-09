import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from tasks.config import cfg
from random import randint, choice


class TrainSampler(torch_sampler.BatchSampler):

    def __init__(self, subdivision, batch_size, max_iterations, num_samples, image_scales, scale_interval):

        if not isinstance(batch_size, _int_classes) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))

        if not isinstance(max_iterations, _int_classes) or max_iterations <= 0 or max_iterations < batch_size:
            raise ValueError("max_iterations should be a positive integeral value, "
                             "but got max_iterations={}".format(max_iterations))

        if not isinstance(subdivision, _int_classes) or subdivision <= 0 :
            raise ValueError("subdivision should be a positive integeral value, "
                             "but got subdivision={}".format(subdivision))

        if not isinstance(num_samples, _int_classes) or num_samples <= 0 :
            raise ValueError("num_samples should be a positive integeral value, "
                             "but got num_samples={}".format(num_samples))

        self.subdivision    = subdivision 
        self.batch_size     = batch_size 
        self.max_iterations = max_iterations
        self.num_samples    = num_samples
        self.image_scales   = image_scales
        self.scale_interval = scale_interval

    def __iter__(self):
        batch = []
        
        minibatch_size = (self.batch_size/self.subdivision)
        image_pos = 0
        scale_counter = 0

        hflip_options = [False]
        if cfg.TEST.BBOX_AUG.H_FLIP:
            hflip_options.append(True)

        # Starts with the smallest resolution
        actual_scale = self.image_scales[0]

        for i in range(self.max_iterations):
            image_idx = randint(0, self.num_samples-1)
            
            batch.append((image_idx, actual_scale, choice(hflip_options), image_pos))
            image_pos += 1

            if image_pos == minibatch_size:
                yield batch
                batch = []
                image_pos = 0
                scale_counter += 1

                if len(self.image_scales) > 1 and self.scale_interval*(self.batch_size/minibatch_size) == scale_counter:

                    new_scale = choice(self.image_scales)
                    while new_scale == actual_scale:
                        new_scale = choice(self.image_scales)

                    # print("Changing scale from",actual_scale,"to", new_scale)
                    actual_scale = new_scale
                    
                    scale_counter = 0
        
    def __len__(self):
        return self.max_iterations


class TestSampler(torch_sampler.BatchSampler):
    def __init__(self, num_images):
        self.num_images = num_images

    def __iter__(self):
        batch = []
        
        hflip_options = [False]
        if cfg.TEST.BBOX_AUG.H_FLIP:
            hflip_options.append(True)

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            for hflip in hflip_options:
                for image_idx in range(self.num_images):
                    
                    batch.append((image_idx, scale, hflip, 0))
                    yield batch
                    batch = []
    
    def __len__(self):
        return self.num_images *  len(cfg.TEST.BBOX_AUG.SCALES) * (cfg.TEST.BBOX_AUG.H_FLIP*2)


class VisualizeSampler(torch_sampler.BatchSampler):
    def __init__(self, num_images):
        self.num_images = num_images

    def __iter__(self):
        batch = []
        
        for image_idx in range(self.num_images):
            
            batch.append((image_idx, -1, False, 0))
            yield batch
            batch = []
    
    def __len__(self):
        return self.num_images
        

def collate_minibatch(batch):

    data   = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    rois   = torch.cat([item[2] for item in batch])

    img_key             = [item[3] for item in batch]
    original_proposals  = [item[4] for item in batch]

    return data, target, rois, img_key, original_proposals


def collate_minibatch_visualize(batch):

    data   = [item[0] for item in batch]
    target = [item[1] for item in batch]
    rois   = [item[2] for item in batch]

    img_key             = [item[3] for item in batch]
    original_proposals  = [item[4] for item in batch]

    return data, target, rois, img_key, original_proposals






