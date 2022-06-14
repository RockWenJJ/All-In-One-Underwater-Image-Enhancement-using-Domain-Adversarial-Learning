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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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

def test(fE, fI, fN, dataloader, model_name):
    cls_out = open(os.path.join('./results', model_name, 'results.txt'), 'w')
    cls_out.write("name\tclass\tssim\tpsnr\n")
    cls_count = [0, 0, 0, 0, 0, 0]
    # real_cls_count = [0, 0, 0, 0, 0, 0]
    ssim_scores = []
    psnr_scores = []
    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, name = data
        uw_img = Variable(uw_img).cuda()
        
        try:
            fE_out, enc_outs = fE(uw_img)
            fI_out = to_img(fI(fE_out, enc_outs).detach())
    
            fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            
            
            if fN is not None:
                cls = int(F.softmax(fN(fE_out), dim=1).max(1)[1])
                cls_count[cls] += 1
            else:
                cls = 0
                
            if cl_img is not None:
                cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
                ssim_score = ssim(fI_out, cl_img, multichannel=True)
                psnr_score = psnr(cl_img, fI_out)
                
            else:
                ssim_score, psnr_score = 0., 0.

            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
            cls_out.write('%s\t%d\t%.3f\t%.3f\n'%(name[0], cls, ssim_score, psnr_score))
                # real_cls = int(name[0].split('_')[1])
                # real_cls_count[real_cls] += 1
            
            fI_out = cv2.cvtColor(fI_out, cv2.COLOR_RGB2BGR)
            if name[0].endswith('.png') or name[0].endswith('jpg'):
                cv2.imwrite(os.path.join('./results/', model_name, name[0]), fI_out)
            else:
                cv2.imwrite(os.path.join('./results/', model_name, name[0]+'.png'), fI_out)
        except Exception as e:
            print(e)
    cls_out.write("Average SSIM: %.3f, Aeverage PSNR: %.3f\n"%(sum(ssim_scores)/len(ssim_scores), sum(psnr_scores)/len(psnr_scores)))
    cls_out.close()
    print(cls_count)
    print("Average SSIM: %.3f, Aeverage PSNR: %.3f\n"%(sum(ssim_scores)/len(ssim_scores), sum(psnr_scores)/len(psnr_scores)))
    # print(real_cls_count)

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default', help='Path of training input data')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of input image channels')
    parser.add_argument('--data_path', type=str, default=None, help='Path of training input data')
    parser.add_argument('--label_path', type=str, default=None, help='Path of reference input data')
    parser.add_argument('--test_size', type=int, default=3000, help='Lambda for N loss')
    parser.add_argument('--fe_load_path', type=str, default=None, help='Load path for pretrained fN')
    parser.add_argument('--fi_load_path', type=str, default=None, help='Load path for pretrained fE')
    parser.add_argument('--load_nuisance', type=int, default=0, help='whether to load nuisance classifier')
    parser.add_argument('--fn_load_path', type=str, default=None, help='Load path for pretrained nuisance classifier')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of water types')
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

    if args.load_nuisance:
        nuisance = Classifier(args.num_classes).cuda()
        nuisance.load_state_dict(torch.load(args.fn_load_path))
    else:
        nuisance = None

    fE.eval()
    fI.eval()
    nuisance.eval()
    
    test_dataset = UIEBDataset(args.data_path)
    # test_dataset = NYUUWDataset(args.data_path, args.label_path)#UIEBDataset(args.data_path)
    
    if args.load_nuisance:
        test_dataset.transform = transforms.Compose([
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])

    batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test(fE, fI, nuisance, dataloader, args.name)

if __name__== "__main__":
    main()