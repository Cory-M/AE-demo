import argparse
import os
import shutil
import time
import yaml
import random
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import models
import cv2
from tensorboardX import SummaryWriter
import logging
from torch.autograd import Variable
from memcached_dataset import McDataset, McDataset_path
from utils import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, DistributedGivenIterationSampler, simple_group_split
from XXXXX import dist_init, average_gradients, DistModule
from util_acc import get_accuracy
from torchvision.utils import save_image

from sklearn.cluster import KMeans

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Reconstruction Training')
parser.add_argument('--config', default='/cfgs/config.yaml')
parser.add_argument('--load-ckpt', default='', type=str)
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('-a', '--augmentation', action='store_true')
parser.add_argument('--bm-acc', action='store_true')
parser.add_argument('-c','--cls-acc', action='store_true')
parser.add_argument('--load-iter', default='', type=str)
parser.add_argument('--acc-result', default='acc_result/', type=str)
parser.add_argument('--iteration', default=10000, type=int)


def main():
	global args, best_prec1
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f)

	for k, v in config['common'].items():
		setattr(args, k, v)
	for k, v in config['ckpt'].items():
		setattr(args, k, v)
	rank, world_size = dist_init(args.port)
	
	# create model
	print("=> creating model '{}'".format(args.encoder))

	image_size = 32
	input_size = 32
	if args.bm_acc:	
		model = models.__dict__[args.encoder](config['model'],mode='hidden',evaluation=True,normalize=False)
	if args.cls_acc:
		model = models.__dict__[args.encoder](config['model'],mode='itm',evaluation=True, normalize=False)

	model.cuda()
	model = DistModule(model)
	model.eval()

	cudnn.benchmark = True

	# Data loading
	if args.augmentation:
		_transform = transforms.Compose([
			  transforms.ToTensor()])
	else:
		_transform = transforms.ToTensor()

	if args.cls_acc:
		train_dataset = McDataset(
			args.train_root,
			args.train_source,
			_transform
			)
	val_dataset = McDataset(
		args.val_root,
		args.val_source,
		_transform
		)

	if args.cls_acc:
		train_sampler = DistributedGivenIterationSampler(train_dataset, args.iteration, args.batch_size, last_iter=-1)
	if args.bm_acc:
		train_sampler = DistributedSampler(train_dataset)

	val_sampler = DistributedSampler(val_dataset)

	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False, sampler=train_sampler)

	val_loader = DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False, sampler=val_sampler)
	
	os.makedirs(args.load_ckpt.split('ckpt')[0]+'eval',exist_ok=True)

	if args.bm_acc:
		kmeans = KMeans(n_clusters=10)
		bm_save = args.load_ckpt.split('ckpt')[0]+'eval/best_map_acc.txt'
		img_save = args.load_ckpt.split('ckpt')[0]+'eval/'
		bm_acc(train_loader, model, args.load_ckpt, bm_save, img_save, kmeans)
	if args.cls_acc:
		cls_acc_save = args.load_ckpt.split('ckpt')[0]+'eval/cls_acc.txt'
		classification_acc(train_loader, val_loader, model, args.load_ckpt, config, cls_acc_save)


def bm_acc(train_loader, model, load_ckpt, save_path, img_save, kmeans):
	
	load_state(load_ckpt, model, evaluation=True)
	model.eval()
	rank = dist.get_rank()

	feature = []
	labels = []
	paths = []
	for i, (data, target, path) in enumerate(train_loader):
		fea = model(Variable(data.type(torch.cuda.FloatTensor)))
		for label in target.numpy():
			labels.append(label)
		for x in fea.cpu().data.numpy():
			feature.append(x.tolist())
		for y in path:
			paths.append(y)
		
	pred_label = kmeans.fit_predict(np.array(feature))
	true_label = np.asarray(labels)
	bm_acc = get_accuracy(pred_label,true_label,10)[0]
	label_dict = dict()
	for i, x in enumerate(pred_label):
		if x in label_dict:
			label_dict[x].append(i)
		else:
			label_dict[x] = [i]
	img_matrix = np.zeros((32*10+2*9, 32*30+2*29,3))

	for i, cls in enumerate(label_dict):
		random.shuffle(label_dict[cls])
		for j in range(30):
			img_matrix[i*33:(i+1)*33-1,j*33:(j+1)*33-1] = cv2.imread(paths[label_dict[cls][j]])
	
	if rank == 0:
		print(bm_acc)
		load_iter = load_ckpt.split('_')[-1].split('.')[0]
		cv2.imwrite(img_save+str(load_iter)+'.png', img_matrix)
		with open(save_path, "a") as fout:
			fout.write("iteration: {}\tbm_acc: {}\n".format(load_iter, bm_acc))
 
def classification_acc(train_loader, val_loader, model, load_ckpt, option, save_path):
	
	load_state(load_ckpt, model, evaluation=True)
	model.eval()
	world_size = dist.get_world_size()
	rank = dist.get_rank()
	
	classifier = nn.Linear(option['model']['intermediate_size'], 10)
	classifier.cuda()
	classifier = DistModule(classifier)
	
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

	losses = AverageMeter(10)
	top1 = AverageMeter(10)
	evaluation_result_list = list()
	
	for i,(input, target) in enumerate(train_loader):
		target = target.cuda(async=True)
		input = input.cuda()
		
		input_var = torch.autograd.Variable(input, requires_grad=True)
		target_var = torch.autograd.Variable(target)

		feature = model(input_var)
		
		prediction = classifier(feature)

		loss = criterion(prediction, target_var) / world_size
		prec1 = accuracy(prediction.data, target)[0]

		reduced_loss = loss.data.clone()
		reduced_prec1 = prec1.clone() / world_size

		dist.all_reduce(reduced_loss)
		dist.all_reduce(reduced_prec1)

		losses.update(reduced_loss[0])
		top1.update(reduced_prec1[0])

		optimizer.zero_grad()
		loss.backward()
		average_gradients(classifier)
		optimizer.step()

		if i % args.print_freq == 0 and rank == 0:
			print('Iter: [{0}/{1}]\t''Loss {loss.val:.4f}\t''acc {top1.val:.4f}'.format(
					i, len(train_loader), loss=losses, top1=top1))
		
		if i % 1000 == 0:
			test_prec = test_acc(val_loader, model, classifier)
			evaluation_result_list.append(test_prec)
				
	# output acc to file
	if rank == 0:
		max_evaluation_acc = max(evaluation_result_list)
		load_iter = load_ckpt.split('_')[-1].split('.')[0]
		with open(save_path, "a") as fout:
			fout.write("iteration: {}\tacc: {}\n".format(load_iter, max_evaluation_acc))

def test_acc(val_loader, model, classifier):
	classifier.eval()
	top1 = AverageMeter(0)
	rank = dist.get_rank()
	world_size = dist.get_world_size()

	for k, (data, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(data.cuda(), requires_grad=True)
		target_var = torch.autograd.Variable(target.cuda())

		feature = model(input_var)
		prediction = classifier(feature)
		prec = accuracy(prediction.data, target)[0]

		reduce_prec1 = prec.clone()
		top1.update(reduce_prec1[0])
		
		if k % args.print_freq == 0 and rank == 0:
			print('Test: [{0}/{1}]\t'
				  'top1_acc {top1.val:.4f}'.format(k, len(val_loader),top1=top1))
	if rank == 0:
		print('Test_acc_top1 {top1.avg:.3f}'.format(top1=top1))
	
	classifier.train()
	
	return top1.avg

if __name__ == '__main__':
	main()
