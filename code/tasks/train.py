import os
import sys
import argparse
import traceback
import logging
import pickle
import yaml
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths

from tasks.config    import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb  import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch

import utils.misc as misc_utils
from utils.optimizer_handler import OptimizerHandler
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats

from models import *

from pdb import set_trace as pause

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)


def parse_args():
	"""Parse input arguments"""
	parser = argparse.ArgumentParser(description='Train a X-RCNN network')

	parser.add_argument('--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
	parser.add_argument('--set', dest='set_cfgs', help='Set config keys. Key value sequence seperate by whitespace.e.g. [key] [value] [key] [value]', default=[], nargs='+')
	parser.add_argument('--disp_interval', help='Display training info every N iterations', default=20, type=int)

	# Optimization
	# These options has the highest prioity and can overwrite the values in config file
	# or values set by set_cfgs. `None` means do not overwrite.
	parser.add_argument('--bs', dest='batch_size', help='Explicitly specify to overwrite the value comed from cfg_file.', type=int, default=1)

	parser.add_argument('--nw', dest='num_workers', help='Explicitly specify to overwrite number of workers to load data. Defaults to 4', type=int)
	parser.add_argument('--iter_size', help='Update once every iter_size steps, as in Caffe.', type=int)

	parser.add_argument('--o', dest='optimizer', help='Training optimizer.', default=None)
	parser.add_argument('--lr', help='Base learning rate.', default=None, type=float)
	parser.add_argument('--lr_decay_gamma', help='Learning rate decay rate.', default=None, type=float)

	# Epoch
	parser.add_argument('--start_step', help='Starting step count for training epoch. 0-indexed.', default=0, type=int)

	# Resume training: requires same iterations per epoch
	parser.add_argument('--resume', help='resume to training on a checkpoint', action='store_true')
	parser.add_argument('--no_save', help='do not save anything', action='store_true')
	parser.add_argument('--load_ckpt', help='checkpoint path to load')
	parser.add_argument('--use_tfboard', help='Use tensorflow tensorboard to log training info', type=bool, default=True)
	parser.add_argument('--model', help='Set model', type=str)
	parser.add_argument('--fixed_seed', help='Set model', type=bool, default=False)

	return parser.parse_args()


def save_ckpt(output_dir, args, step, train_size, model, optimizer, is_bkp=False):
	"""Save checkpoint"""
	if args.no_save:
		return
	ckpt_dir = os.path.join(output_dir, 'ckpt')
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	if is_bkp:
		save_name = os.path.join(ckpt_dir, 'bkp.pth')
	else:
		save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
	
	model_state_dict = model.state_dict()
	torch.save({
		'step': step,
		'train_size': train_size,
		'batch_size': args.batch_size,
		'model': model.state_dict(),
		'optimizer': optimizer.optimizer.state_dict()}, save_name)
   
	logger.info('save model: %s', save_name)


def save_training_config(output_dir, cfg, args):
	args.cfg_filename = os.path.basename(args.cfg_file)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	blob = {'cfg': yaml.dump(cfg), 'args': args}
	with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
		pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)


def load_ckpt(load, model, optimizer):
	if load:
		load_name = args.load_ckpt
		logging.info("loading checkpoint %s", load_name)
		checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
		net_utils.load_ckpt(model, checkpoint['model'])
		if args.resume:
			args.start_step = checkpoint['step'] + 1
			if 'train_size' in checkpoint:  # For backward compatibility
				if checkpoint['train_size'] != train_size:
					print('train_size value: %d different from the one in checkpoint: %d'
						  % (train_size, checkpoint['train_size']))


			optimizer.load_state_dict(checkpoint['optimizer'])

		del checkpoint
		torch.cuda.empty_cache()


def check_and_overwrite_params(cfg, args):

	### Overwrite some solver settings from command line arguments
	if args.batch_size is None:
		args.batch_size = cfg.TRAIN.IMS_PER_BATCH
	
	if args.num_workers is not None:
		cfg.DATA_LOADER.NUM_THREADS = args.num_workers
	
	if args.optimizer is not None:
		cfg.SOLVER.TYPE = args.optimizer
	
	if args.lr is not None:
		cfg.SOLVER.BASE_LR = args.lr
	
	if args.lr_decay_gamma is not None:
		cfg.SOLVER.GAMMA = args.lr_decay_gamma
	
	if args.start_step is not None:
		cfg.START_STEP = args.start_step

	if args.iter_size is not None:
		cfg.TRAIN.ITERATION_SIZE = args.iter_size

	assert (args.batch_size % cfg.NUM_GPUS) == 0, 'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)

	### Adjust learning based on batch size change linearly
	# For iter_size > 1, gradients are `accumulated`, so lr is scaled based
	# on batch_size instead of effective_batch_size
	old_base_lr = cfg.SOLVER.BASE_LR
	cfg.SOLVER.BASE_LR *= args.batch_size / cfg.TRAIN.IMS_PER_BATCH
	print('Adjust BASE_LR linearly according to batch_size change:\n'
		  '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

	### Adjust solver steps
	step_scale = cfg.TRAIN.IMS_PER_BATCH / cfg.TRAIN.ITERATION_SIZE
	old_solver_steps = cfg.SOLVER.STEPS
	old_max_iter = cfg.SOLVER.MAX_ITER
	cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
	cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
	print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
		  '    SOLVER.STEPS: {} --> {}\n'
		  '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
												  old_max_iter, cfg.SOLVER.MAX_ITER))

def main():

	if not torch.cuda.is_available():
		sys.exit("Need a CUDA device to run the training code, sry bro :(.")
	else:
		cfg.CUDA = True
		cfg.NUM_GPUS = torch.cuda.device_count()


	#######~~~.Parameters stuff.~~~#######
	args = parse_args()
	print('Called with args:\n', args)

	# Enables fixed seed
	if args.fixed_seed:
		np.random.seed(cfg.RNG_SEED)
		torch.manual_seed(cfg.RNG_SEED)
		if cfg.CUDA:
		    torch.cuda.manual_seed_all(cfg.RNG_SEED)
	torch.backends.cudnn.deterministic = True


	cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	check_and_overwrite_params(cfg, args)
	assert_and_infer_cfg()

	#indentificantion of a specific running
	args.run_name = misc_utils.get_run_name() + '_step'
	output_dir = misc_utils.get_output_dir(args)

	save_training_config(output_dir, cfg, args)
	
	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		tblogger = SummaryWriter(output_dir)
	else:
		tblogger = None


	#######~~~.Dataset.~~~#######
	timers = defaultdict(Timer)

	# Roi datasets
	timers['roidb'].tic()
	roidb, ratio_list, ratio_index = combined_roidb_for_training(
		cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
	timers['roidb'].toc()
	roidb_size = len(roidb)
	logger.info('{:d} roidb entries'.format(roidb_size))
	logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)


	batchSampler = BatchSampler(
		sampler=MinibatchSampler(ratio_list, ratio_index),
		batch_size=args.batch_size,
		drop_last=True
	)
	dataset = RoiDataLoader(
		roidb,
		cfg.MODEL.NUM_CLASSES,
		training=True)
	
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_sampler=batchSampler,
		num_workers=cfg.DATA_LOADER.NUM_THREADS,
		collate_fn=collate_minibatch)
	dataiterator = iter(dataloader)


	########~~~.Model.~~~#######
	model = eval(args.model).loot_model(args)

	if cfg.CUDA:
		model.cuda()

	optimizer = OptimizerHandler(model, cfg)

	load_ckpt(args.load_ckpt, model, optimizer)


	#######~~~.Training Loop.~~~#######
	try:
		# Effective training sample size for one epoch
		train_size = roidb_size // args.batch_size * args.batch_size
		CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / (cfg.NUM_GPUS * cfg.TRAIN.ITERATION_SIZE))

		training_stats = TrainingStats(args, tblogger)

		model.train()
		
		logger.info('Training starts !')
		step = args.start_step
		for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

			optimizer.update_learning_rate(step)

			training_stats.IterTic()
			optimizer.zero_grad()
			for inner_iter in range(cfg.TRAIN.ITERATION_SIZE):
				try:
					input_data = next(dataiterator)
				except StopIteration:
					dataiterator = iter(dataloader)
					input_data = next(dataiterator)

				for key in input_data:
					if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
						input_data[key] = list(map(Variable, input_data[key]))

				# input_data['inner_iter'] = torch.tensor((inner_iter))
				model.set_inner_iter(step)

				im_data = input_data['data'][0].cuda()
				rois    = input_data['rois'][0].cuda().type(im_data.dtype)
				labels  = input_data['labels'][0].cuda().type(im_data.dtype)

				net_outputs = model(im_data, rois, labels)

				training_stats.UpdateIterStats(net_outputs, inner_iter)
				loss = net_outputs['total_loss']
				loss.backward(retain_graph=True)
			
			optimizer.step()
			training_stats.IterToc()

			training_stats.LogIterStats(step, optimizer.get_lr())

			if (step+1) % CHECKPOINT_PERIOD == 0:
				save_ckpt(output_dir, args, step, train_size, model, optimizer)

		# Training ends, saves the last checkpoint
		save_ckpt(output_dir, args, step, train_size, model, optimizer)

	except (RuntimeError, KeyboardInterrupt):
		del dataiterator
		logger.info('Save ckpt on exception ...')
		save_ckpt(output_dir, args, step, train_size, model, optimizer)
		logger.info('Save ckpt done.')
		stack_trace = traceback.format_exc()
		print()
		print(stack_trace)

	finally:
		if args.use_tfboard and not args.no_save:
			tblogger.close()

if __name__ == '__main__':
	main()
