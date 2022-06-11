import os
import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import numpy as np
from utils.ssim_pytorch import ssim_torch

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
    
    return sum(ssim_scores) / len(dataloader), sum(psnr_scores) / len(dataloader), sum(mse_scores) / len(
        dataloader), corr / len(dataloader)


def compute_prw_val_metrics(prwnet, dataloader):
    """
        Compute SSIM, PSNR scores for the validation set
    """
    
    prwnet.eval()
    
    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    corr = 0
    
    criterion_MSE = nn.MSELoss().cuda()
    
    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, _ = data
        uw_img = Variable(uw_img).cuda()
        cl_img = Variable(cl_img, requires_grad=False).cuda()
        
        outs = prwnet(uw_img)
        
        mse_scores.append(criterion_MSE(outs[0], cl_img).item())
        
        fI_out = (outs[0] * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        
        ssim_scores.append(ssim(fI_out, cl_img, multichannel=True))
        psnr_scores.append(psnr(cl_img, fI_out))
    
    prwnet.train()
    
    return sum(ssim_scores) / len(dataloader), sum(psnr_scores) / len(dataloader), sum(mse_scores) / len(
        dataloader)

def backward_reconstruction_loss(decoder, encoder_out, encoder_outs, cl_img, criterion_MSE, optimizer_decoder, reconstruct_loss_weight, retain_graph):
    """
        Backpropagate the reconstruction loss
    """
    
    decoder_out = to_img(decoder(encoder_out, encoder_outs))
    mse_loss = criterion_MSE(decoder_out, cl_img) * reconstruct_loss_weight
    ssim_loss = (1 - ssim_torch(decoder_out, cl_img)) * reconstruct_loss_weight / 10.
    optimizer_decoder.zero_grad()
    decoder_loss = mse_loss + ssim_loss
    decoder_loss.backward(retain_graph=retain_graph)
    optimizer_decoder.step()
    
    return decoder_out, decoder_loss


def backward_nuisance_loss(nuisance_classifier, encoder_out, nuisance_target, criterion_CE, optimizer_nuisance, nuisance_loss_weight):
    """
        Backpropagate the nuisance loss
    """
    
    nuisance_out = nuisance_classifier(encoder_out.detach())
    nuisance_loss = criterion_CE(nuisance_out, nuisance_target) * nuisance_loss_weight
    optimizer_nuisance.zero_grad()
    nuisance_loss.backward()
    optimizer_nuisance.step()
    
    return nuisance_loss


def backward_adv_loss(nuisance_classifier, encoder_out, adv_loss_weight, num_classes, neg_entropy):
    """
        Backpropagate the adversarial loss
    """
    
    nuisance_out = nuisance_classifier(encoder_out)
    adv_loss = calc_adv_loss(nuisance_out, num_classes, neg_entropy) * adv_loss_weight
    adv_loss.backward()
    
    return adv_loss


def write_to_log(log_file_path, status):
    """
        Write to the log file
    """
    
    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')


def calc_adv_loss(nuisance_out, num_classes, neg_entropy):
    """
        Calculate the adversarial loss (negative entropy or cross entropy with uniform distribution)
    """
    
    if neg_entropy:
        fN_out_softmax = F.softmax(nuisance_out, dim=1)
        return torch.mean(torch.sum(fN_out_softmax * torch.log(torch.clamp(fN_out_softmax, min=1e-10, max=1.0)), 1))
    else:
        fN_out_log_softmax = F.log_softmax(nuisance_out, dim=1)
        return -torch.mean(torch.div(torch.sum(fN_out_log_softmax, 1), num_classes))