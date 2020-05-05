import init_paths
import argparse
import os
import sys
import pickle
import traceback

import numpy as np
import yaml
import torch

from models import *
from datasets.pascal_voc import VOCDetection
from tasks.config    import cfg, load_config

import utils.optimizer_utils as optimizer_utils
import utils.misc   as misc_utils
import utils.data_transforms as dt_trans

from utils.data_samplers           import TrainSampler, collate_minibatch
from utils.detectron_weight_helper import load_detectron_weight
from utils.training_stats import TrainingStats


import logging
from utils.logging import setup_logging

from pdb import set_trace as pause

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)



def parse_args():
	"""Parse input arguments"""
	parser = argparse.ArgumentParser(description='Train a WSOD detection network')

	parser.add_argument('--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
	parser.add_argument('--set', dest='set_cfgs', help='Set config keys. Key value sequence seperate by whitespace.e.g. [key] [value] [key] [value]', default=[], nargs='+')
	parser.add_argument('--disp_interval', help='Display training info every N iterations', default=20, type=int)

	# Optimization
	# These options has the highest prioity and can overwrite the values in config file
	# or values set by set_cfgs. `None` means do not overwrite.
	parser.add_argument('--bs', dest='batch_size', help='Explicitly specify to overwrite the value comed from cfg_file.', type=int, default=1)

	parser.add_argument('--nw', dest='num_workers', help='Explicitly specify to overwrite number of workers to load data. Defaults to 4', type=int, default=4)
	parser.add_argument('--iter_size', help='Update once every iter_size steps, as in Caffe.', type=int, default=4)

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

	# if isinstance(model, mynn.DataParallel):
	#     model = model.module
	model_state_dict = model.state_dict()
	torch.save({
		'step': step,
		'train_size': train_size,
		'batch_size': args.batch_size,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict()}, save_name)
   
	logger.info('save model: %s', save_name)


def main():
	"""Main function"""

	if not torch.cuda.is_available():
		sys.exit("Need a CUDA device to run the code.")

	args = parse_args()
	print('Called with args:')
	print(args)


	load_config(args.cfg_file)

	if cfg.NUM_GPUS > 0:
		cfg.CUDA = True
	else:
		raise ValueError("Need Cuda device to run !")

	

	### Adaptively adjust some configs ###
	original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
	original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
	original_num_gpus = cfg.NUM_GPUS
	if args.batch_size is None:
		args.batch_size = original_batch_size
	cfg.NUM_GPUS = torch.cuda.device_count()

	assert (args.batch_size % cfg.NUM_GPUS) == 0, \
		'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
	cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
	effective_batch_size = args.iter_size * args.batch_size
	print('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))

	print('Adaptive config changes:')
	print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
	print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
	print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

	### Adjust learning based on batch size change linearly
	# For iter_size > 1, gradients are `accumulated`, so lr is scaled based
	# on batch_size instead of effective_batch_size
	old_base_lr = cfg.SOLVER.BASE_LR
	cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
	print('Adjust BASE_LR linearly according to batch_size change:\n'
		  '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

	### Adjust solver steps
	step_scale = original_batch_size / effective_batch_size
	old_solver_steps = cfg.SOLVER.STEPS
	old_max_iter = cfg.SOLVER.MAX_ITER
	cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
	cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
	print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
		  '    SOLVER.STEPS: {} --> {}\n'
		  '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
												  old_max_iter, cfg.SOLVER.MAX_ITER))

	if args.num_workers is not None:
		cfg.DATA_LOADER.NUM_THREADS = args.num_workers
	print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

	### Overwrite some solver settings from command line arguments
	if args.optimizer is not None:
		cfg.SOLVER.TYPE = args.optimizer
	if args.lr is not None:
		cfg.SOLVER.BASE_LR = args.lr
	if args.lr_decay_gamma is not None:
		cfg.SOLVER.GAMMA = args.lr_decay_gamma
	
  


	# np.random.seed(cfg.RNG_SEED)
	# torch.manual_seed(cfg.RNG_SEED)
	# if cfg.CUDA:
	#     torch.cuda.manual_seed_all(cfg.RNG_SEED)
	torch.backends.cudnn.deterministic = True


	transforms = dt_trans.Compose([
		dt_trans.Normalize([102.9801, 115.9465, 122.7717]),
		dt_trans.HorizontalFlip(),
		dt_trans.Resize(),
		dt_trans.ToTensor(),
		
		])


	if cfg.TRAIN.DATASET == 'voc_2007_trainval':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year='2007', image_set='trainval', transforms=transforms) 
	
	elif cfg.TRAIN.DATASET == 'voc_2012_trainval':
		dataset =  VOCDetection(cfg.DATA_DIR + '/', year='2012', image_set='trainval', transforms=transforms)

	# Effective training sample size for one epoch
	train_size = len(dataset) // args.batch_size * args.batch_size



	batchSampler = TrainSampler(
		subdivision    = cfg.TRAIN.ITERATION_SIZE,
		batch_size     = cfg.TRAIN.ITERATION_SIZE,
		max_iterations = cfg.SOLVER.MAX_ITER,
		num_samples    = len(dataset),
		image_scales   = cfg.TRAIN.SCALES,
		scale_interval = 10
	)

	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_sampler=batchSampler,
		num_workers=cfg.DATA_LOADER.NUM_THREADS,
		collate_fn=collate_minibatch)
	dataiterator = iter(dataloader)

	### Model ###
	model = eval(args.model).loot_model(args)

	if cfg.CUDA:
		model.cuda()

	### Optimizer ###
	bias_params = []
	bias_param_names = []
	nonbias_params = []
	nonbias_param_names = []
	nograd_param_names = []
	for key, value in model.named_parameters():
		if value.requires_grad:
			if 'bias' in key:
				bias_params.append(value)
				bias_param_names.append(key)
			else:
				nonbias_params.append(value)
				nonbias_param_names.append(key)
		else:
			nograd_param_names.append(key)

	# Learning rate of 0 is a dummy value to be set properly at the start of training
	params = [
		{'params': nonbias_params,
		 'lr': 0,
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
		{'params': bias_params,
		 'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
	]
	# names of paramerters for each paramter
	param_names = [nonbias_param_names, bias_param_names]

	if cfg.SOLVER.TYPE == "SGD":
		optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
	elif cfg.SOLVER.TYPE == "Adam":
		optimizer = torch.optim.Adam(params)

	### Load checkpoint
	if args.load_ckpt:
		load_name = args.load_ckpt
		logging.info("loading checkpoint %s", load_name)
		checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
		optimizer_utils.load_ckpt(model, checkpoint['model'])
		if args.resume:
			args.start_step = checkpoint['step'] + 1
			if 'train_size' in checkpoint:  # For backward compatibility
				if checkpoint['train_size'] != train_size:
					print('train_size value: %d different from the one in checkpoint: %d'
						  % (train_size, checkpoint['train_size']))

			# reorder the params in optimizer checkpoint's params_groups if needed
			# misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

			# There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
			# However it's fixed on master.
			optimizer.load_state_dict(checkpoint['optimizer'])
			# misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
		del checkpoint
		torch.cuda.empty_cache()


	lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.


	### Training Setups ###
	args.run_name = misc_utils.get_run_name() + '_step'
	output_dir = misc_utils.get_output_dir(args)

	args.cfg_filename = os.path.basename(args.cfg_file)

	if not args.no_save:
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		blob = {'cfg': yaml.dump(cfg), 'args': args}
		with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
			pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

		if args.use_tfboard:
			from tensorboardX import SummaryWriter
			# Set the Tensorboard logger
			tblogger = SummaryWriter(output_dir)

	### Training Loop ###
	model.train()

	CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / (cfg.NUM_GPUS * args.iter_size))

	# Set index for decay steps
	decay_steps_ind = None
	for i in range(1, len(cfg.SOLVER.STEPS)):
		if cfg.SOLVER.STEPS[i] >= args.start_step:
			decay_steps_ind = i
			break
	if decay_steps_ind is None:
		decay_steps_ind = len(cfg.SOLVER.STEPS)

	training_stats = TrainingStats(
		args,
		args.disp_interval,
		tblogger if args.use_tfboard and not args.no_save else None)
	try:
		logger.info('Training starts !')
		step = args.start_step
		for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

			# Warm up
			if step < cfg.SOLVER.WARM_UP_ITERS:
				method = cfg.SOLVER.WARM_UP_METHOD
				if method == 'constant':
					warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
				elif method == 'linear':
					alpha = step / cfg.SOLVER.WARM_UP_ITERS
					warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
				else:
					raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
				lr_new = cfg.SOLVER.BASE_LR * warmup_factor
				optimizer_utils.update_learning_rate(optimizer, lr, lr_new)
				lr = optimizer.param_groups[0]['lr']
				assert lr == lr_new
			elif step == cfg.SOLVER.WARM_UP_ITERS:
				optimizer_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
				lr = optimizer.param_groups[0]['lr']
				assert lr == cfg.SOLVER.BASE_LR

			# Learning rate decay
			if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
					step == cfg.SOLVER.STEPS[decay_steps_ind]:
				logger.info('Decay the learning on step %d', step)
				lr_new = lr * cfg.SOLVER.GAMMA
				optimizer_utils.update_learning_rate(optimizer, lr, lr_new)
				lr = optimizer.param_groups[0]['lr']
				assert lr == lr_new
				decay_steps_ind += 1

			training_stats.IterTic()
			optimizer.zero_grad()
			for inner_iter in range(args.iter_size):
				try:
					input_data = next(dataiterator)
				except StopIteration:
					dataiterator = iter(dataloader)
					input_data = next(dataiterator)

				model.set_inner_iter(step)

				im_data = input_data[0].cuda()
				labels  = input_data[1].cuda()
				rois    = input_data[2].cuda()

				net_outputs = model(im_data, rois, labels)
			  

				training_stats.UpdateIterStats(net_outputs, inner_iter)
				loss = net_outputs['total_loss']
				loss.backward(retain_graph=True)
			optimizer.step()
			training_stats.IterToc()

			training_stats.LogIterStats(step, lr)

			if (step+1) % CHECKPOINT_PERIOD == 0:
				save_ckpt(output_dir, args, step, train_size, model, optimizer)

		# ---- Training ends ----
		# Save last checkpoint
		save_ckpt(output_dir, args, step, train_size, model, optimizer)

	except (RuntimeError, KeyboardInterrupt):
		del dataiterator
		logger.info('Save ckpt on exception ...')
		save_ckpt(output_dir, args, step, train_size, model, optimizer)
		logger.info('Save ckpt done.')
		stack_trace = traceback.format_exc()
		print(stack_trace)

	finally:
		if args.use_tfboard and not args.no_save:
			tblogger.close()


if __name__ == '__main__':
	main()
