import os
import logging
import shutil
import torch
from datetime import datetime
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import numpy as np
import torch.nn as nn

def simple_group_split(world_size, rank, num_groups):
	groups = []
	rank_list = np.split(np.arange(world_size), num_groups)
	rank_list = [list(map(int, x)) for x in rank_list]
	for i in range(num_groups):
		groups.append(dist.new_group(ranks=rank_list[i]))
	group_size = world_size // num_groups
	return groups[rank//group_size]

def create_logger(name, log_file, level=logging.INFO):
	l = logging.getLogger(name)
	formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
	fh = logging.FileHandler(log_file)
	fh.setFormatter(formatter)
	sh = logging.StreamHandler()
	sh.setFormatter(formatter)
	l.setLevel(level)
	l.addHandler(fh)
	l.addHandler(sh)
	return l

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, length=0):
		self.length = length
		self.reset()

	def reset(self):
		if self.length > 0:
			self.history = []
		else:
			self.count = 0
			self.sum = 0.0
		self.val = 0.0
		self.avg = 0.0

	def update(self, val):
		if self.length > 0:
			self.history.append(val)
			if len(self.history) > self.length:
				del self.history[0]

			self.val = self.history[-1]
			self.avg = np.mean(self.history)
		else:
			self.val = val
			self.sum += val
			self.count += 1
			self.avg = self.sum / self.count
			
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

class IterLRScheduler(object):
	def __init__(self, optimizer, milestones, lr_mults, last_iter=-1):
		assert len(milestones) == len(lr_mults), "{} vs {}".format(milestone, lr_mults)
		self.milestones = milestones
		self.lr_mults = lr_mults
		if not isinstance(optimizer, torch.optim.Optimizer):
			raise TypeError('{} is not an Optimizer'.format(
				type(optimizer).__name__))
		self.optimizer = optimizer
		for i, group in enumerate(optimizer.param_groups):
			if 'lr' not in group:
				raise KeyError("param 'lr' is not specified "
							   "in param_groups[{}] when resuming an optimizer".format(i))
		self.last_iter = last_iter

	def _get_lr(self):
		try:
			pos = self.milestones.index(self.last_iter)
		except ValueError:
			return list(map(lambda group: group['lr'], self.optimizer.param_groups))
		except:
			raise Exception('wtf?')
		return list(map(lambda group: group['lr']*self.lr_mults[pos], self.optimizer.param_groups))

	def get_lr(self):
		return list(map(lambda group: group['lr'], self.optimizer.param_groups))

	def step(self, this_iter=None):
		if this_iter is None:
			this_iter = self.last_iter + 1
		self.last_iter = this_iter
		for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
			param_group['lr'] = lr


def save_checkpoint(state, is_best, filename):
	torch.save(state, filename+'.pth.tar')
	if is_best:
		shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')

def load_state(path, model, optimizer=None, evaluation=False):
	def map_func(storage, location):
		return storage.cuda()
	if os.path.isfile(path):
		print("=> loading checkpoint '{}'".format(path))
		checkpoint = torch.load(path, map_location=map_func)
		if evaluation:
			model.load_state_dict(checkpoint['encoder'], strict=False)
			ckpt_keys = set(checkpoint['encoder'].keys())
			own_keys = set(model.state_dict().keys())
		else:
			model[0].load_state_dict(checkpoint['encoder'], strict=False)
			model[1].load_state_dict(checkpoint['decoder'], strict=False)
			model[2].load_state_dict(checkpoint['discriminator'], strict=False)
			ckpt_keys = set(checkpoint['encoder'].keys()) | set(checkpoint['decoder'].keys()) | set(checkpoint['discriminator'].keys())
			own_keys = set(model[0].state_dict().keys()) | set(model[1].state_dict().keys()) | set(model[2].state_dict().keys())
		missing_keys = own_keys - ckpt_keys
		for k in missing_keys:
			print('caution: missing keys from checkpoint {}: {}'.format(path, k))

		if optimizer != None:
			#best_prec1 = checkpoint['best_prec1']
			last_iter = checkpoint['step']
			optimizer[0].load_state_dict(checkpoint['optimizer_G'])
			optimizer[1].load_state_dict(checkpoint['optimizer_D'])
			print("=> also loaded optimizer from checkpoint '{}' (iter {})"
				  .format(path, last_iter))
			return last_iter
	else:
		print("=> no checkpoint found at '{}'".format(path))



