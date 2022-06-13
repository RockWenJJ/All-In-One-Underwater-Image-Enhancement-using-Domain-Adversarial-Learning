import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from PIL import Image
from dataset.dataset import NYUUWDataset
from tqdm import tqdm, trange
import random
from torchvision import models
import numpy as np
from models.networks import Classifier, UNetEncoder, UNetDecoder
from models.prwnet import PRWNet
from utils.utils import *
import datetime
import argparse
import wandb
from torch.utils.tensorboard import SummaryWriter

def config_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='default', help='Path of training input data')
	parser.add_argument('--data_path', type=str, default=None, help='Path of training input data')
	parser.add_argument('--label_path', type=str, default=None, help='Path of training label data')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
	parser.add_argument('--save_interval', type=int, default=5, help='Save models after this many epochs')
	parser.add_argument('--start_epoch', type=int, default=0, help='Start training from this epoch')
	parser.add_argument('--end_epoch', type=int, default=200, help='Train till this epoch')
	parser.add_argument('--num_classes', type=int, default=6, help='Number of water types')
	parser.add_argument('--num_channels', type=int, default=3, help='Number of input image channels')
	parser.add_argument('--train_size', type=int, default=30000, help='Size of the training dataset')
	parser.add_argument('--test_size', type=int, default=3000, help='Size of the testing dataset')
	parser.add_argument('--val_size', type=int, default=3000, help='Size of the validation dataset')
	parser.add_argument('--load_path', type=str, default=None, help='Load path for pretrained model')
	parser.add_argument('--reconstruction_loss_weight', type=float, default=100.0, help='Lambda for I loss')
	parser.add_argument('--wandb', type=int, default=0, help='whether to use wandb for logging or not')
	return parser

def main():
	parser = config_parser()
	args = parser.parse_args()

	# Define datasets and dataloaders
	train_path = os.path.join(args.data_path, 'train')
	test_path = os.path.join(args.data_path, 'test')
	val_path = os.path.join(args.data_path, 'val')
	train_dataset = NYUUWDataset(train_path, args.label_path, mode='train')
	val_dataset = NYUUWDataset(test_path, args.label_path, mode='val')
	test_dataset = NYUUWDataset(val_path, args.label_path, mode='test')

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)
	val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8)
	
	# Define models, criterion and optimizers
	prw_net = PRWNet().to('cuda')
	
	if args.load_path is not None:
		prw_net.load_state_dict(torch.load(os.path.join(args.load_path, 'prwnet_%d.pth'%args.start_epoch)))
	
	optimizer = torch.optim.Adam(prw_net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
	criterion_mse = nn.MSELoss().cuda()

	prw_net.train()
	
	out_path = os.path.join("./checkpoints", args.name)
	os.makedirs(out_path, exist_ok=True)
	log_file_path = os.path.join(out_path, 'log_file.txt')

	now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

	status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
	write_to_log(log_file_path, status)
	if args.wandb:
		wandb_logger = wandb.init(project='UIE', config=args, name=args.name)
	logger = SummaryWriter(os.path.join(out_path))

	val_ssim = -1. # initial SSIM and nuisance classification accuracy
	
	epoch = args.start_epoch
	steps_per_epoch = len(train_dataset)/args.batch_size
	total_step = int(epoch * steps_per_epoch)
	
	# main training loop when encoder-decoder works fine
	ave_mse_loss, mse_loss_count = 0., 0
	ave_ssim_loss, ssim_loss_count = 0., 0
	for epoch in range(args.start_epoch, args.end_epoch+1):
		status = 'Average decoder out SSIM: {}\n'.format(val_ssim)
		print(status)
		write_to_log(log_file_path, status)
		
		for idx, data in tqdm(enumerate(train_dataloader)):
			uw_img, cl_img, water_type, _ = data
			uw_img = Variable(uw_img).cuda()
			cl_img = Variable(cl_img, requires_grad=False).cuda()
			
			outs = prw_net(uw_img)
			
			optimizer.zero_grad()
			# calculates output losses, including mse loss and ssim loss
			mse_loss, ssim_loss = 0., 0.
			for i in range(len(outs)):
				mse_loss0 = criterion_mse(outs[i], cl_img) * args.reconstruction_loss_weight
				ssim_loss0 = (1 - ssim_torch(outs[i], cl_img)) * args.reconstruction_loss_weight / 10.
				mse_loss += mse_loss0
				ssim_loss += ssim_loss0
			total_loss = mse_loss + ssim_loss
			ave_mse_loss += mse_loss.item()
			ave_ssim_loss += ssim_loss.item()
			total_loss.backward()
			optimizer.step()
			
			if total_step % 100 == 0 and total_step != 0:
				print("Epoch: {}, Iter: {}, Total Step: {}, mse loss: {}, ssim loss: {}, total loss: {}".format(
					epoch, idx, total_step, ave_mse_loss/100., ave_ssim_loss/100., (ave_mse_loss+ave_ssim_loss)/100.))
				if args.wandb:
					wandb_logger.log({"train/mse_loss":ave_mse_loss/100.,
									  "train/ssim_loss":ave_ssim_loss/100.,
									  "train/total_loss":(ave_mse_loss+ave_ssim_loss)/100.}, total_step)
				logger.add_scalar("train/mse_loss", ave_mse_loss/100., global_step=total_step)
				logger.add_scalar("train/ssim_loss", ave_ssim_loss/100., global_step=total_step)
				logger.add_scalar("train/total_loss", (ave_mse_loss+ave_ssim_loss)/100., global_step=total_step)
				ave_mse_loss, ave_ssim_loss = 0., 0.
			
			if total_step % 1000 == 0 and args.wandb:
				out0_images = wandb.Image(outs[0].cpu().data)
				out1_images = wandb.Image(outs[1].cpu().data)
				uw_images = wandb.Image(uw_img.cpu().data)
				cl_images = wandb.Image(cl_img.cpu().data)
				save_dict = {"train/out0_images": out0_images,
							 "train/out1_images": out1_images,
							 "train/uw_images": uw_images,
							 "train/cl_images": cl_images}
				
				wandb_logger.log(save_dict, total_step)
			
			total_step += 1
		

		if epoch % args.save_interval == 0:
			torch.save(prw_net.state_dict(), './checkpoints/{}/prwnet_{}.pth'.format(args.name, epoch))
		
		# Evaluate

		val_ssim, val_psnr, val_mse = compute_prw_val_metrics(prw_net, val_dataloader)
		if args.wandb:
			wandb_logger.log({"val/ssim": val_ssim,
					   "val/psnr": val_psnr,
					   "val/mse": val_mse}, total_step)
			
		# log with tensorboard
		logger.add_scalar("val/ssim", val_ssim, global_step=total_step)
		logger.add_scalar("val/psnr", val_psnr, global_step=total_step)
		logger.add_scalar("val/mse", val_mse, global_step=total_step)

if __name__== "__main__":
	main()