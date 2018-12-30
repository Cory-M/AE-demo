import argparse
import os
import shutil
import time
import yaml
import itertools
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
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import logging
import pdb
from utils import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, DistributedGivenIterationSampler, simple_group_split
from XXXXX import dist_init, average_gradients, DistModule
from data_factory import gaussian_mixture, gaussian_mixture_axis
from torchvision.utils import save_image

from sklearn.cluster import KMeans

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Reconstruction Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--resume-opt', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('-v','--vae', action='store_true')
parser.add_argument('--m-loss', action='store_true')

Tensor = torch.cuda.FloatTensor

class ColorAugmentation(object):
	def __init__(self, eig_vec=None, eig_val=None):
		if eig_vec == None:
			eig_vec = torch.Tensor([
				[ 0.4009,  0.7192, -0.5675],
				[-0.8140, -0.0045, -0.5808],
				[ 0.4203, -0.6948, -0.5836],
			])
		if eig_val == None:
			eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
		self.eig_val = eig_val	# 1*3
		self.eig_vec = eig_vec	# 3*3

	def __call__(self, tensor):
		assert tensor.size(0) == 3
		alpha = torch.normal(means=torch.zeros_like(self.eig_val))*0.1
		quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
		tensor = tensor + quatity.view(3, 1, 1)
		return tensor

def main():
	global args, best_prec1
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f)

	for k, v in config['common'].items():
		setattr(args, k, v)
	for k, v in config['model'].items():
		setattr(args, k, v)
	
	rank, world_size = dist_init(args.port)
	
	if rank == 0:
		os.makedirs(config['ckpt']['save_path'], exist_ok=True)
		os.makedirs(config['ckpt']['save_path']+'/images', exist_ok=True)
		os.makedirs(config['ckpt']['save_path']+'/ckpt', exist_ok=True)
		os.makedirs(config['ckpt']['save_path']+'/npy', exist_ok=True)
		os.makedirs(config['ckpt']['save_path']+'/eval', exist_ok=True)
	# create model
	print("=> creating encoder '{}'".format(args.encoder))
	print("=> creating decoder '{}'".format(args.decoder))

	image_size = 32
	input_size = 32
	
	encoder = models.__dict__[args.encoder](config['model'])
	decoder = models.__dict__[args.decoder](config['model'])
	discriminator = models.__dict__[args.discriminator](config['model'])

	encoder.cuda()
	encoder = DistModule(encoder)
	decoder.cuda()
	decoder = DistModule(decoder)
	discriminator.cuda()
	discriminator = DistModule(discriminator)

	# define loss function (criterion) and optimizer
	adv_criterion = torch.nn.BCELoss()
	recon_criterion = torch.nn.SmoothL1Loss()
	aug_criterion = torch.nn.CosineEmbeddingLoss()

	optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()),
		  lr=config['model']['lr_G'], betas=(config['model']['b1'], config['model']['b2']))
	optimizer_D = torch.optim.Adam(discriminator.parameters(),
		  lr=config['model']['lr_D'], betas=(config['model']['b1'], config['model']['b2']))


	# optionally resume from a checkpoint
	last_iter = -1
	best_prec1 = 0
	if args.load_path:
		model = [encoder, decoder, discriminator]
		optimizer = [optimizer_G, optimizer_D]
		last_iter = load_state(args.load_path, model, optimizer=optimizer, evaluation=False)

	cudnn.benchmark = True

	# Data loading code
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	_transform = transforms.Compose([
		  transforms.Resize(input_size),
		  transforms.ToTensor(),
#		  normalize,
		  ])
	aug_transform = transforms.Compose([
		  transforms.RandomResizedCrop(input_size),
		  transforms.RandomHorizontalFlip(),
		  transforms.ToTensor(),
		  ColorAugmentation(),
#		  normalize,
		  ])

	train_dataset = McDataset_aug(args.train_root, args.train_source, _transform, aug_transform)
	
	train_sampler = DistributedGivenIterationSampler(train_dataset, args.max_iter,
		  args.batch_size, last_iter=last_iter)
	
	test_sampler = DistributedSampler(train_dataset)
		  
	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False, sampler=train_sampler)

	test_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False, sampler=test_sampler)

	if rank == 0:
		tb_logger = SummaryWriter(config['ckpt']['save_path'])
		logger = create_logger('global_logger', config['ckpt']['save_path']+'/log.txt')
		logger.info('{}'.format(args))
		logger.info('{}'.format(config['model']))
	else:
		tb_logger = None

	train(train_loader, test_loader, encoder, decoder, discriminator, adv_criterion, recon_criterion, aug_criterion, optimizer_G, optimizer_D, last_iter+1, tb_logger, config)

def train(train_loader, test_loader, encoder, decoder, discriminator, adv_criterion,
		  recon_criterion, aug_criterion, optimizer_G, optimizer_D, start_iter, tb_logger, config):

	batch_time = AverageMeter(10)
	data_time = AverageMeter(10)

	recon_losses = AverageMeter(10)
	adv_losses = AverageMeter(10)
	aug_losses = AverageMeter(10)
	D_loss = AverageMeter(10)

	# switch to train mode
	encoder.train()
	decoder.train()
	discriminator.train()

	world_size = dist.get_world_size()
	rank = dist.get_rank()

	logger = logging.getLogger('global_logger')

	end = time.time()

	for i, (data, aug_data, target) in enumerate(train_loader):
		curr_step = start_iter + i	
		data_time.update(time.time() - end)

		real_imgs = Variable(data.type(Tensor))
		aug_imgs = Variable(aug_data.type(Tensor))

		valid = Variable(Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

		# ------------------
		# Train Generator, aka AutoEncoder-Decoder
		# ------------------

		optimizer_G.zero_grad()

		encoded_imgs, itm_vec, mu, logvar = encoder(real_imgs)
		encoded_aug_imgs, _, _, _ = encoder(aug_imgs)
		decoded_imgs = decoder(encoded_imgs)

		#pdb.set_trace()
		adv_loss = adv_criterion(discriminator(encoded_imgs), valid)
		recon_loss = recon_criterion(decoded_imgs, real_imgs)
		aug_loss = aug_criterion(encoded_imgs, encoded_aug_imgs, valid)
		g_loss = (config['model']['adv_coeff'] * adv_loss + 
			  config['model']['recon_coeff'] * recon_loss + 
			  config['model']['aug_coeff'] * aug_loss) / world_size

		reduced_g_loss = g_loss.data.clone()
		dist.all_reduce(reduced_g_loss)
		
		recon_losses.update(recon_loss.data.clone()[0])
		adv_losses.update(adv_loss.data.clone()[0])
		aug_losses.update(aug_loss.data.clone()[0])

		g_loss.backward()
		average_gradients(encoder)
		average_gradients(decoder)
		optimizer_G.step()

		# -------------------
		# Train Discriminator
		# -------------------
		optimizer_D.zero_grad()

		if config['model']['constrain'] == 'intermediate':
			z = Variable(Tensor(gaussian_mixture_axis(args.batch_size,
					  n_dim=config['model']['intermediate_size'],shift=config['model']['shift'])))
		if config['model']['constrain'] == 'hidden':
			z = Variable(Tensor(gaussian_mixture_axis(args.batch_size,
					  n_dim=config['model']['hidden_size'],shift=config['model']['shift'])))

		real_loss = adv_criterion(discriminator(z), valid)
		if config['model']['constrain'] == 'intermediate':
			fake_loss = adv_criterion(discriminator(itm_vec.detach()), fake)
		if config['model']['constrain'] == 'hidden':
			fake_loss = adv_criterion(discriminator(encoded_imgs.detach()),fake)
		d_loss = (0.5 * (real_loss + fake_loss)) / world_size

		reduced_d_loss = d_loss.data.clone()
		dist.all_reduce(reduced_d_loss)
		D_loss.update(reduced_d_loss[0])
		
		d_loss.backward()
		average_gradients(discriminator)
		optimizer_D.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if curr_step % config['ckpt']['print_freq'] == 0 and rank == 0:
			tb_logger.add_scalar('recon_loss', recon_losses.avg, curr_step)
			tb_logger.add_scalar('adv_loss', adv_losses.avg, curr_step)
			tb_logger.add_scalar('aug_loss', aug_losses.avg, curr_step)
			tb_logger.add_scalar('D_loss', D_loss.avg, curr_step)

			logger.info('Iter: [{0}/{1}]\t'
				  'adv_Loss {adv_losses.val:.4f} ({adv_losses.avg:.4f})\t'
				  'aug_Los {aug_losses.val:.4f} ({aug_losses.avg:.4f})\t'
				  'recon_Loss {recon_losses.val:.4f} ({recon_losses.avg:.4f})\t'
				  'D_loss {D_loss.val:.4f} ({D_loss.avg:.4f})\t'
				  'lr_G {lr_G:.4f}\t'
				  'lr_D {lr_D:.4f}\t'.format(
				   curr_step, len(train_loader),
				   D_loss=D_loss, adv_losses=adv_losses, aug_losses=aug_losses, 
				   recon_losses=recon_losses, lr_G=config['model']['lr_G'],lr_D=config['model']['lr_D']))
		
		if curr_step % config['ckpt']['save_freq'] == 0 and rank == 0:
			if config['ckpt']['save_ckpt']:
				save_checkpoint({
					  'step': curr_step,
					  'encoder': encoder.state_dict(),
					  'decoder': decoder.state_dict(),
					  'discriminator': discriminator.state_dict(),
					  'optimizer_G':optimizer_G.state_dict(), 
					  'optimizer_D':optimizer_D.state_dict()},
					  False, config['ckpt']['save_path'] + '/ckpt/ckpt_' + str(curr_step))
			if config['ckpt']['save_img']:
				n = min(data.size(0),8)
				real_pic = torch.Tensor(data.cpu()[:n])
				fake_pic = torch.Tensor(decoded_imgs.cpu().data.numpy()[:n])
				save_image(torch.cat([real_pic,fake_pic]),
					  config['ckpt']['save_path']+'/images/'+str(curr_step)+'.png',nrow=n)
			if config['ckpt']['save_feature']:
				encoder.eval()
				metadata = []
				hidden_vec = []
				itm_vecs = []
				for i, (data, _, target) in enumerate(test_loader):
					encoded_imgs, itm_vec, mu, logvar = encoder(Variable(data.type(Tensor)))
					for label in target.numpy():
						metadata.append(label)
					for x in encoded_imgs.cpu().data.numpy():
						hidden_vec.append(x.tolist())
					for x in itm_vec.cpu().data.numpy():
					  	itm_vecs.append(x.tolist())
				array = np.concatenate((np.array(metadata).reshape(-1,1), 
							 np.array(hidden_vec),np.array(itm_vecs)), axis=1)
				np.save(config['ckpt']['save_path'] + '/npy/'
					  + str(config['model']['adv_coeff']) + '_' 
					  + str(config['model']['recon_coeff']) + '_s'
					  + str(config['model']['shift']) + '_'
					  + str(curr_step)+'.npy', array)
				#matrix = torch.FloatTensor(np.array(matrix))
				#tb_logger.add_embedding(matrix, metadata=metadata, global_step=curr_step)
				encoder.train()


if __name__ == '__main__':
	main()
