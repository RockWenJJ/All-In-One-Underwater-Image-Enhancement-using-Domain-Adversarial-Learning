import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from PIL import Image
from dataset.dataset import NYUUWDataset
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
from models.networks import Classifier, UNetEncoder, UNetDecoder
import click
import datetime
import argparse
import wandb

def to_img(x):
	"""
		Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
	"""

	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 3, 256, 256)

	return x

def set_requires_grad(nets, requires_grad=False):
	"""
		Make parameters of the given network not trainable
	"""

	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad

	return requires_grad

def compute_val_metrics(fE, fI, fN, dataloader, no_adv_loss):
	"""
		Compute SSIM, PSNR scores for the validation set
	"""

	fE.eval()
	fI.eval()
	fN.eval()

	mse_scores = []
	ssim_scores = []
	psnr_scores = []
	corr = 0

	criterion_MSE = nn.MSELoss().cuda()

	for idx, data in tqdm(enumerate(dataloader)):
		uw_img, cl_img, water_type, _ = data
		uw_img = Variable(uw_img).cuda()
		cl_img = Variable(cl_img, requires_grad=False).cuda()
		
		fE_out, enc_outs = fE(uw_img)
		fI_out = to_img(fI(fE_out, enc_outs))
		fN_out = F.softmax(fN(fE_out), dim=1)

		if int(fN_out.max(1)[1].item()) == int(water_type.item()):
			corr += 1

		mse_scores.append(criterion_MSE(fI_out, cl_img).item())

		fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
		cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

		ssim_scores.append(ssim(fI_out, cl_img, multichannel=True))
		psnr_scores.append(psnr(cl_img, fI_out))

	fE.train()
	fI.train()
	if not no_adv_loss:
		fN.train()

	return sum(ssim_scores)/len(dataloader), sum(psnr_scores)/len(dataloader), sum(mse_scores)/len(dataloader), corr/len(dataloader)

def backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI, lambda_I_loss, retain_graph):
	"""
		Backpropagate the reconstruction loss
	"""

	fI_out = to_img(fI(fE_out, enc_outs))
	I_loss = criterion_MSE(fI_out, cl_img) * lambda_I_loss
	optimizer_fI.zero_grad()
	I_loss.backward(retain_graph=retain_graph)
	optimizer_fI.step()

	return fI_out, I_loss

def backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss):
	"""
		Backpropagate the nuisance loss
	"""

	fN_out = fN(fE_out.detach())
	N_loss = criterion_CE(fN_out, actual_target) * lambda_N_loss
	optimizer_fN.zero_grad()
	N_loss.backward()
	optimizer_fN.step()

	return N_loss

def backward_adv_loss(fN, fE_out, lambda_adv_loss, num_classes, neg_entropy):
	"""
		Backpropagate the adversarial loss
	"""

	fN_out = fN(fE_out)
	adv_loss = calc_adv_loss(fN_out, num_classes, neg_entropy) * lambda_adv_loss
	adv_loss.backward()

	return adv_loss

def write_to_log(log_file_path, status):
	"""
		Write to the log file
	"""

	with open(log_file_path, "a") as log_file:
		log_file.write(status+'\n')

def calc_adv_loss(fN_out, num_classes, neg_entropy):
	"""
		Calculate the adversarial loss (negative entropy or cross entropy with uniform distribution)
	"""

	if neg_entropy:
		fN_out_softmax = F.softmax(fN_out, dim=1)
		return torch.mean(torch.sum(fN_out_softmax * torch.log(torch.clamp(fN_out_softmax, min=1e-10, max=1.0)), 1))
	else:
		fN_out_log_softmax = F.log_softmax(fN_out, dim=1)
		return -torch.mean(torch.div(torch.sum(fN_out_log_softmax, 1), num_classes))

def config_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='default', help='Path of training input data')
	parser.add_argument('--data_path', type=str, default=None, help='Path of training input data')
	parser.add_argument('--label_path', type=str, default=None, help='Path of training label data')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
	parser.add_argument('--save_interval', type=int, default=5, help='Save models after this many epochs')
	parser.add_argument('--start_epoch', type=int, default=1, help='Start training from this epoch')
	parser.add_argument('--end_epoch', type=int, default=200, help='Train till this epoch')
	parser.add_argument('--num_classes', type=int, default=6, help='Number of water types')
	parser.add_argument('--num_channels', type=int, default=3, help='Number of input image channels')
	parser.add_argument('--train_size', type=int, default=30000, help='Size of the training dataset')
	parser.add_argument('--test_size', type=int, default=3000, help='Size of the testing dataset')
	parser.add_argument('--val_size', type=int, default=3000, help='Size of the validation dataset')
	parser.add_argument('--fe_load_path', type=str, default=None, help='Load path for pretrained fN')
	parser.add_argument('--fi_load_path', type=str, default=None, help='Load path for pretrained fE')
	parser.add_argument('--fn_load_path', type=str, default=None, help='Load path for pretrained fN')
	parser.add_argument('--lambda_i_loss', type=float, default=100.0, help='Lambda for I loss')
	parser.add_argument('--lambda_n_loss',type=float, default=1.0, help='Lambda for N loss')
	parser.add_argument('--lambda_adv_loss', type=float, default=1.0, help='Lambda for adv loss')
	parser.add_argument('--fi_threshold', type=float, default=0.9, help='Train fI till this threshold')
	parser.add_argument('--fn_threshold', type=float, default=0.85, help='Train fN till this threshold')
	parser.add_argument('--continue_train', type=bool, default=False, help='Continue training from start_epoch')
	parser.add_argument('--neg_entropy', type=bool, default=True,
				  help='Use KL divergence instead of cross entropy with uniform distribution')
	parser.add_argument('--no_adv_loss', type=bool, default=False, help='Use adversarial loss during training or not')
	parser.add_argument('--wandb', type=int, default=0, help='whether to use wandb for logging or not')
	return parser

def main():
	parser = config_parser()
	args = parser.parse_args()
	name = args.name
	data_path = args.data_path
	label_path = args.label_path
	learning_rate = args.learning_rate
	batch_size = args.batch_size
	save_interval = args.save_interval
	start_epoch = args.start_epoch
	end_epoch = args.end_epoch
	num_classes = args.num_classes
	num_channels = args.num_channels
	train_size = args.train_size
	test_size = args.test_size
	val_size = args.val_size
	fe_load_path = args.fe_load_path
	fi_load_path = args.fi_load_path
	fn_load_path = args.fn_load_path
	lambda_i_loss = args.lambda_i_loss
	lambda_n_loss = args.lambda_n_loss
	lambda_adv_loss = args.lambda_adv_loss
	fi_threshold = args.fi_threshold
	fn_threshold = args.fn_threshold
	continue_train = args.continue_train
	neg_entropy = args.neg_entropy
	no_adv_loss = args.no_adv_loss

	fE_load_path = fe_load_path
	fI_load_path = fi_load_path
	fN_load_path = fn_load_path

	lambda_I_loss = lambda_i_loss
	lambda_N_loss = lambda_n_loss

	fI_threshold = fi_threshold
	fN_threshold = fn_threshold
	
	wandb_logger = wandb.init(project='UIE', config=args, name=args.name) if args.wandb else None

	# Define datasets and dataloaders
	train_path = os.path.join(data_path, 'train')
	test_path = os.path.join(data_path, 'test')
	val_path = os.path.join(data_path, 'val')
	train_dataset = NYUUWDataset(train_path, label_path, mode='train')
	val_dataset = NYUUWDataset(test_path, label_path, mode='val')
	test_dataset = NYUUWDataset(val_path, label_path, mode='test')

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

	if not no_adv_loss:
		"""
			Define the nuisance classifier to include the adversarial loss in the model
		"""

		fN = Classifier(num_classes).cuda()
		fN_req_grad = True
		fN.train()
		criterion_CE = nn.CrossEntropyLoss().cuda()
		optimizer_fN = torch.optim.Adam(fN.parameters(), lr=learning_rate,
								 weight_decay=1e-5)

	# Define models, criterion and optimizers
	fE = UNetEncoder(num_channels).cuda()
	fI = UNetDecoder(num_channels).cuda()

	criterion_MSE = nn.MSELoss().cuda()

	optimizer_fE = torch.optim.Adam(fE.parameters(), lr=learning_rate,
								 weight_decay=1e-5)
	optimizer_fI = torch.optim.Adam(fI.parameters(), lr=learning_rate,
								 weight_decay=1e-5)

	fE.train()
	fI.train()

	if continue_train:
		"""
			Load pretrained models to continue training
		"""

		if fE_load_path:
			fE.load_state_dict(torch.load(fE_load_path))
			print ('Loaded fE from {}'.format(fE_load_path))
		if fI_load_path:
			fI.load_state_dict(torch.load(fI_load_path))
			print ('Loaded fI from {}'.format(fI_load_path))
		if not no_adv_loss and fN_load_path:
			fN.load_state_dict(torch.load(fN_load_path))
			print ('Loaded fN from {}'.format(fN_load_path))

	if not os.path.exists('./checkpoints/{}'.format(name)):
		os.mkdir('./checkpoints/{}'.format(name))

	log_file_path = './checkpoints/{}/log_file.txt'.format(name)

	now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

	status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
	write_to_log(log_file_path, status)

	# Compute the initial cross validation scores
	if continue_train and not no_adv_loss:
		fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss)
	else:
		fI_val_ssim = -1
		fN_val_acc = -1

	# Train only the encoder-decoder upto a certain threshold
	total_step = 0
	while fI_val_ssim < fI_threshold and not continue_train:
		epoch = start_epoch

		status = 'Avg fI val SSIM: {}, Avg fN val acc: {}\nCurrent fI threshold: {}, Current fN threshold: {}'.format(fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold)
		print (status)
		write_to_log(log_file_path, status)

		for idx, data in tqdm(enumerate(train_dataloader)):
			uw_img, cl_img, water_type, _ = data
			uw_img = Variable(uw_img).cuda()
			cl_img = Variable(cl_img, requires_grad=False).cuda()
			
			fE_out, enc_outs = fE(uw_img)
			optimizer_fE.zero_grad()
			fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI, lambda_I_loss, retain_graph=not no_adv_loss)

			progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

			optimizer_fE.step()
			
			total_step += 1
			if idx % 50 == 0:
				print(progress)

			if total_step % 100 == 0:
				fi_images = wandb.Image(fI_out.cpu().data)
				uw_images = wandb.Image(uw_img.cpu().data)
				cl_images = wandb.Image(cl_img.cpu().data)
				save_dict = {"train/fI_images": fi_images,
							 "train/uw_images": uw_images,
							 "train/cl_images": cl_images,
							 "train/I_loss": I_loss.item()}

				# print (progress)
				wandb.log(save_dict, total_step)

		# torch.save(fE.state_dict(), './checkpoints/{}/fE_latest.pth'.format(name))
		# torch.save(fI.state_dict(), './checkpoints/{}/fI_latest.pth'.format(name))
		if epoch % save_interval == 0:
			torch.save(fE.state_dict(), './checkpoints/{}/fE_{}.pth'.format(name, epoch))
			torch.save(fI.state_dict(), './checkpoints/{}/fI_{}.pth'.format(name, epoch))
		
		# status = 'End of epoch. Models saved.'
		print (status)
		write_to_log(log_file_path, status)
		
		val_ssim, val_psnr, val_mse, val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss)
		wandb.log({"val/ssim": val_ssim,
				   "val/psnr": val_psnr,
				   "val/mse": val_mse,
				   "val/acc": val_acc}, total_step)

	for epoch in range(start_epoch, end_epoch):
		"""
			Main training loop
		"""

		if not no_adv_loss:
			"""
				Print the current cross-validation scores
			"""

			status = 'Avg fI val SSIM: {}, Avg fN val acc: {}\nCurrent fI threshold: {}, Current fN threshold: {}'.format(fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold)
			# print (status)
			# write_to_log(log_file_path, status)

		for idx, data in tqdm(enumerate(train_dataloader)):
			uw_img, cl_img, water_type, _ = data
			uw_img = Variable(uw_img).cuda()
			cl_img = Variable(cl_img, requires_grad=False).cuda()
			actual_target = Variable(water_type, requires_grad=False).cuda()
			
			fE_out, enc_outs = fE(uw_img)
			
			if fI_val_ssim < fI_threshold:
				"""
					Train the encoder-decoder only
				"""

				optimizer_fE.zero_grad()
				fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI, lambda_I_loss, retain_graph=not no_adv_loss)

				if not no_adv_loss:
					if fN_req_grad:
						fN_req_grad = set_requires_grad(fN, requires_grad=False)
					adv_loss = backward_adv_loss(fN, fE_out, lambda_adv_loss, num_classes, neg_entropy)
					progress = "\tEpoch: {}\tIter: {}\tI_loss: {}\tadv_loss: {}".format(epoch, idx, I_loss.item(), adv_loss.item())
				else:
					progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())
					adv_loss = None

				optimizer_fE.step()
				
				if idx % 50 == 0:
					print(progress)
				
				if total_step % 100 == 0:
					fi_images = wandb.Image(fI_out.cpu().data)
					uw_images = wandb.Image(uw_img.cpu().data)
					cl_images = wandb.Image(cl_img.cpu().data)
					save_dict = {"train/fI_images": fi_images,
								 "train/uw_images": uw_images,
								 "train/cl_images": cl_images,
								 "train/I_loss": I_loss.item(),
								 "train/adv_loss": adv_loss.item() if adv_loss else 10.0}
					
					# print (progress)
					wandb.log(save_dict, total_step)

			elif fN_val_acc < fN_threshold:
				"""
					Train the nusiance classifier only
				"""

				if not fN_req_grad:
					fN_req_grad = set_requires_grad(fN, requires_grad=True)

				N_loss = backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss)
				progress = "\tEpoch: {}\tIter: {}\tN_loss: {}".format(epoch, idx, N_loss.item())
				
				if total_step % 100 == 0:
					wandb.log({"train/N_loss": N_loss.item()}, total_step)

			else:
				"""
					Train the encoder-decoder only
				"""

				optimizer_fE.zero_grad()
				fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI, lambda_I_loss, retain_graph=not no_adv_loss)

				if not no_adv_loss:
					if fN_req_grad:
						fN_req_grad = set_requires_grad(fN, requires_grad=False)
					adv_loss = backward_adv_loss(fN, fE_out, lambda_adv_loss, num_classes, neg_entropy)

					progress = "\tEpoch: {}\tIter: {}\tI_loss: {}\tadv_loss: {}".format(epoch, idx, I_loss.item(), adv_loss.item())
				
				else:
					progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

				optimizer_fE.step()
				
				if idx % 50 == 0:
					print(progress)
				
				if total_step % 100 == 0:
					fi_images = wandb.Image(fI_out.cpu().data)
					uw_images = wandb.Image(uw_img.cpu().data)
					cl_images = wandb.Image(cl_img.cpu().data)
					save_dict = {"train/fI_images": fi_images,
								 "train/uw_images": uw_images,
								 "train/cl_images": cl_images,
								 "train/I_loss": I_loss.item(),
								 "train/adv_loss": adv_loss.item()
								 }
					
					# print (progress)
					wandb.log(save_dict, total_step)
			total_step += 1
		

		# # Save models
		# torch.save(fE.state_dict(), './checkpoints/{}/fE_latest.pth'.format(name))
		# torch.save(fI.state_dict(), './checkpoints/{}/fI_latest.pth'.format(name))
		# if not no_adv_loss:
		# 	torch.save(fN.state_dict(), './che"ckpoints/{}/fN_latest.pth'.format(name))

		if epoch % save_interval == 0:
			torch.save(fE.state_dict(), './checkpoints/{}/fE_{}.pth'.format(name, epoch))
			torch.save(fI.state_dict(), './checkpoints/{}/fI_{}.pth'.format(name, epoch))
			if not no_adv_loss:
				torch.save(fN.state_dict(), './checkpoints/{}/fN_{}.pth'.format(name, epoch))

		status = 'End of epoch. Models saved.'
		print (status)
		write_to_log(log_file_path, status)

		if not no_adv_loss:
			"""
				Compute the cross validation scores after the epoch
			"""
			val_ssim, val_psnr, val_mse, val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss)
			wandb.log({"val/ssim": val_ssim,
					   "val/psnr": val_psnr,
					   "val/mse": val_mse,
					   "val/acc": val_acc}, total_step)

if __name__== "__main__":
	# if not os.path.exists('./results'):
	# 	os.mkdir('./results')
	if not os.path.exists('./checkpoints'):
		os.mkdir('./checkpoints')

	main()