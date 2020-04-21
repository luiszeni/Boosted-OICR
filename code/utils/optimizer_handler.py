import torch
import utils.net as net_utils

from pdb import set_trace as pause

class OptimizerHandler():

	def __init__(self, model, cfg):
		super().__init__()

		self.model = model
		self.cfg = cfg
		
		params = self.get_parammeters()

		if self.cfg.SOLVER.TYPE == "SGD":
			self.optimizer = torch.optim.SGD(params, momentum=self.cfg.SOLVER.MOMENTUM)
		elif self.cfg.SOLVER.TYPE == "Adam":
			self.optimizer = torch.optim.Adam(params)

		self.lr = self.optimizer.param_groups[0]['lr'] 

		# Set index for decay steps
		decay_steps_ind = None
		
		for i in range(1, len(self.cfg.SOLVER.STEPS)):
		    if self.cfg.SOLVER.STEPS[i] >=  self.cfg.START_STEP:
		        decay_steps_ind = i
		        break
		if decay_steps_ind is None:
		    decay_steps_ind = len(self.cfg.SOLVER.STEPS)
		self.decay_steps_ind = decay_steps_ind


	def get_parammeters(self):
		### Optimizer ###
		bias_params = []
		bias_param_names = []
		nonbias_params = []
		nonbias_param_names = []
		nograd_param_names = []
		for key, value in self.model.named_parameters():
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
		return [{'params': nonbias_params,
			'lr': 0,
			'weight_decay': self.cfg.SOLVER.WEIGHT_DECAY},
			{'params': bias_params,
			'lr': 0 * (self.cfg.SOLVER.BIAS_DOUBLE_LR + 1),
			'weight_decay': self.cfg.SOLVER.WEIGHT_DECAY if self.cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
			]


	def update_learning_rate(self, step):
		# Warm up
		if step < self.cfg.SOLVER.WARM_UP_ITERS:
		    method = self.cfg.SOLVER.WARM_UP_METHOD
		    if method == 'constant':
		        warmup_factor = self.cfg.SOLVER.WARM_UP_FACTOR
		    elif method == 'linear':
		        alpha = step / self.cfg.SOLVER.WARM_UP_ITERS
		        warmup_factor = self.cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
		    else:
		        raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
		    lr_new = self.cfg.SOLVER.BASE_LR * warmup_factor
		    net_utils.update_learning_rate(self.optimizer, self.lr, lr_new)
		    self.lr = self.optimizer.param_groups[0]['lr']
		    assert self.lr == lr_new
		elif step == self.cfg.SOLVER.WARM_UP_ITERS:
		    net_utils.update_learning_rate(self.optimizer, self.lr, self.cfg.SOLVER.BASE_LR)
		    self.lr = self.optimizer.param_groups[0]['lr']
		    assert self.lr == self.cfg.SOLVER.BASE_LR

		# Learning rate decay
		if self.decay_steps_ind < len(self.cfg.SOLVER.STEPS) and step == self.cfg.SOLVER.STEPS[self.decay_steps_ind]:
		    logger.info('Decay the learning on step %d', step)
		    lr_new = self.lr * self.cfg.SOLVER.GAMMA
		    net_utils.update_learning_rate(self.optimizer, self.lr, lr_new)
		    self.lr = self.optimizer.param_groups[0]['lr']
		    assert self.lr == lr_new
		    self.decay_steps_ind += 1


	def step(self):
		self.optimizer.step()


	def zero_grad(self):
		self.optimizer.zero_grad()


	def get_lr(self):
		return self.lr