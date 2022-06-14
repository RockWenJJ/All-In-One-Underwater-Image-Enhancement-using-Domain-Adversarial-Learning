import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image

class NYUUWDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', mode='train'):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.size = len(self.uw_images)
        self.cl_images = []
        
        for img in self.uw_images:
            cl_image_name = os.path.join(self.label_path, os.path.basename(img).split('_')[0] + '.' + img_format)
            self.cl_images.append(cl_image_name)
        

        self.transform = transforms.Compose([
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            cl_img = self.transform(Image.open(self.cl_images[index]))
            water_type = int(os.path.basename(self.uw_images[index]).split('_')[1])
            name = os.path.basename(self.uw_images[index])[:-4]

            return uw_img, cl_img, water_type, name             

    def __len__(self):
        return self.size


class UIEBDataset2(Dataset):
    def __init__(self, data_path, label_path, img_format='png', mode='train'):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        
        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.size = len(self.uw_images)
        self.cl_images = []
        
        for img in self.uw_images:
            cl_image_name = os.path.join(self.label_path, os.path.basename(img))
            self.cl_images.append(cl_image_name)
        
        self.transform = transforms.Compose([
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        uw_img = self.transform(Image.open(self.uw_images[index]))
        cl_img = self.transform(Image.open(self.cl_images[index]))
        name = os.path.basename(self.uw_images[index])[:-4]
        
        return uw_img, cl_img, name
    
    def __len__(self):
        return self.size

class UIEBDataset(Dataset):
    def __init__(self, data_path, img_format='png',  mode='train'):
        self.data_path = data_path
        self.mode = mode

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.size = len(self.uw_images)


        self.transform = transforms.Compose([
            # transforms.Resize((270, 360)),
            # transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        uw_img = self.transform(Image.open(self.uw_images[index]))

        return uw_img, None, None, os.path.basename(self.uw_images[index])

    def __len__(self):
        return self.size