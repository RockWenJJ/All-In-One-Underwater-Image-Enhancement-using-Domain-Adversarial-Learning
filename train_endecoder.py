import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from PIL import Image
from dataset.dataset import UIEBDataset2
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
from models.networks import Classifier, UNetEncoder, UNetDecoder
from utils.utils import *
import datetime
import argparse
import wandb
from torch.utils.tensorboard import SummaryWriter


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default', help='Path of training input data')
    parser.add_argument('--data_path', type=str, default=None, help='Path of training input data')
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
    parser.add_argument('--fi_load_path', type=str, default=None, help='Load path for pretrained fE')
    parser.add_argument('--reconstruction_loss_weight', type=float, default=100.0, help='Lambda for I loss')
    parser.add_argument('--fi_threshold', type=float, default=0.95, help='Train encoder-decoder till this threshold')
    parser.add_argument('--continue_train', type=bool, default=False, help='Continue training from start_epoch')
    parser.add_argument('--wandb', type=int, default=0, help='whether to use wandb for logging or not')
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    
    # Define datasets and dataloaders
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'test')
    train_label_path = os.path.join(args.data_path, 'train_ref')
    val_label_path = os.path.join(args.data_path, 'test_ref')
    train_dataset = UIEBDataset2(train_path, train_label_path, mode='train')
    val_dataset = UIEBDataset2(val_path, val_label_path, mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8)
    
    
    # Define models, criterion and optimizers
    encoder = UNetEncoder(args.num_channels).cuda()
    decoder = UNetDecoder(args.num_channels).cuda()
    
    if args.load_path is not None:
        encoder.load_state_dict(torch.load(os.path.join(args.load_path, 'encoder_%d.pth' % args.start_epoch)))
        decoder.load_state_dict(torch.load(os.path.join(args.load_path, 'decoder_%d.pth' % args.start_epoch)))
    
    criterion_MSE = nn.MSELoss().cuda()
    
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    encoder.train()
    decoder.train()
    
    out_path = os.path.join("./checkpoints", args.name)
    os.makedirs(out_path, exist_ok=True)
    log_file_path = os.path.join(out_path, 'log_file.txt')
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
    write_to_log(log_file_path, status)
    if args.wandb:
        wandb_logger = wandb.init(project='UIE', config=args, name=args.name)
    logger = SummaryWriter(os.path.join(out_path))
    
    val_ssim, val_acc = -1., -1.  # initial SSIM and nuisance classification accuracy
    
    epoch = args.start_epoch
    steps_per_epoch = len(train_dataset) / args.batch_size
    total_step = int(epoch * steps_per_epoch)
    # Train only the encoder-decoder until up to a certain threshold
    
    start_epoch = epoch + 1
    # main training loop when encoder-decoder works fine
    for epoch in range(start_epoch, args.end_epoch + 1):
        status = 'Average decoder out SSIM: {}.'.format(val_ssim)
        print(status)
        write_to_log(log_file_path, status)
        
        ave_decoder_loss, decoder_loss_count = 0., 0
        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, _ = data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()
            
            encoder_out, encoder_outs = encoder(uw_img)

            optimizer_encoder.zero_grad()
            decoder_out, decoder_loss = backward_reconstruction_loss(decoder, encoder_out, encoder_outs, cl_img,
                                                                     criterion_MSE, optimizer_decoder,
                                                                     args.reconstruction_loss_weight,
                                                                     retain_graph=False)

            optimizer_encoder.step()

            ave_decoder_loss += decoder_loss.item()
            decoder_loss_count += 1

            if decoder_loss_count % 50 == 0 and decoder_loss_count != 0:
                print("Epoch: {}\tIter: {}\tdecoder loss: {}.".format(epoch, idx, ave_decoder_loss / decoder_loss_count))
                if args.wandb:
                    wandb_logger.log({"train/decoder_loss": ave_decoder_loss / decoder_loss_count}, total_step)
    
                logger.add_scalar("train/decoder_loss", ave_decoder_loss / decoder_loss_count,
                                  global_step=total_step)
                ave_decoder_loss, decoder_loss_count = 0., 0

            if total_step % 1000 == 0 and args.wandb:
                fi_images = wandb.Image(decoder_out.cpu().data)
                uw_images = wandb.Image(uw_img.cpu().data)
                cl_images = wandb.Image(cl_img.cpu().data)
                save_dict = {"train/fI_images": fi_images,
                             "train/uw_images": uw_images,
                             "train/cl_images": cl_images
                             }
                wandb_logger.log(save_dict, total_step)
            
            total_step += 1
        
        if epoch % args.save_interval == 0:
            torch.save(encoder.state_dict(), './checkpoints/{}/encoder_{}.pth'.format(args.name, epoch))
            torch.save(decoder.state_dict(), './checkpoints/{}/decoder_{}.pth'.format(args.name, epoch))
        
        val_ssim, val_psnr, val_mse = compute_val_metrics2(encoder, decoder, val_dataloader)
        if args.wandb:
            wandb_logger.log({"val/ssim": val_ssim,
                              "val/psnr": val_psnr,
                              "val/mse": val_mse}, total_step)
        
        # log with tensorboard
        logger.add_scalar("val/ssim", val_ssim, global_step=total_step)
        logger.add_scalar("val/psnr", val_psnr, global_step=total_step)
        logger.add_scalar("val/mse", val_mse, global_step=total_step)


if __name__ == "__main__":
    main()