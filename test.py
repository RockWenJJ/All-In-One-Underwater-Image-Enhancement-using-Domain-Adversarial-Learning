import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from models.networks import UNetEncoder, UNetDecoder, Classifier
import os
from PIL import Image
from dataset.dataset import *
from tqdm import tqdm
import random
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
# from skimage.measure import compare_ssim as ssim_fn
# from skimage.measure import compare_psnr as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from collections import defaultdict
import click
import argparse
import cv2

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def var_to_img(img):
    return (img * 255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

def test(fE, fI, dataloader, model_name):

    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, name = data
        uw_img = Variable(uw_img).cuda()
        
        try:
            fE_out, enc_outs = fE(uw_img)
            fI_out = to_img(fI(fE_out, enc_outs).detach())
    
            fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            fI_out = cv2.cvtColor(fI_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('./results/', model_name, name[0]), fI_out)
        except Exception as e:
            print(e)

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default', help='Path of training input data')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of input image channels')
    parser.add_argument('--data_path', type=str, default=None, help='Path of training input data')
    parser.add_argument('--test_size', type=int, default=3000, help='Lambda for N loss')
    parser.add_argument('--fe_load_path', type=str, default=None, help='Load path for pretrained fN')
    parser.add_argument('--fi_load_path', type=str, default=None, help='Load path for pretrained fE')
    return parser
    

def main():
    parser = config_parser()
    args = parser.parse_args()
        
    os.makedirs(os.path.join('./results', args.name), exist_ok=True)

    fE_load_path = args.fe_load_path
    fI_load_path = args.fi_load_path

    fE = UNetEncoder(args.num_channels).cuda()
    fI = UNetDecoder(args.num_channels).cuda()
    
    fE.load_state_dict(torch.load(fE_load_path))
    fI.load_state_dict(torch.load(fI_load_path))

    fE.eval()
    fI.eval()
    
    test_dataset = UIEBDataset(args.data_path)

    batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test(fE, fI, dataloader, args.name)

if __name__== "__main__":
    main()