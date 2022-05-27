"""
    Script to synthesize a dataset of 10 Jerlov water types (clubbed into
    6 classes - (1, 3), 5, 7, 9, (I, IA, IB), (II, III)) following Anwar et al. (2018).
    The script augments the dataset by generating 6 images of each class using random parameters,
    thus for every ground truth image, we have corresponding 36 images of different water types.
"""

import numpy as np
import scipy.io
import h5py, random
import matplotlib.pyplot as plt
import imageio
from configargparse import ArgumentParser
from tqdm import tqdm
import click, os
import cv2
import configargparse

def config_parser():
    parser: ArgumentParser = configargparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str,
                        help='path to the images')
    parser.add_argument("--deps_path", type=str,
                        help='path to the ')
    parser.add_argument("--output_path", type=str, default='./logs/',
                        help='where to store synthesized images')
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    
    images = os.listdir(args.imgs_path)
    depths = os.listdir(args.deps_path)
    images.sort()
    depths.sort()

    N_lambda = {"1": [0.875, 0.885, 0.75],
                "3": [0.8, 0.82, 0.71],
                "5": [0.67, 0.73, 0.67],
                "7": [0.5, 0.61, 0.62],
                "9": [0.29, 0.46, 0.55],
                "I": [0.982, 0.961, 0.805],
                "IA": [0.975, 0.955, 0.804],
                "IB": [0.968, 0.95, 0.83],
                "II": [0.94, 0.925, 0.8],
                "III": [0.89, 0.885, 0.75]
                }

    data_path = os.path.join(args.output_path, 'data')
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')
    # label_path = os.path.join(args.output_path, 'label')
    # if not os.path.exists(data_path):
    #     os.mkdir(data_path)
    # if not os.path.exists(label_path):
    #     os.mkdir(label_path)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    

    rand = {"1": 3,
            "3": 3,
            "5": 6,
            "7": 6,
            "9": 6,
            "I": 2,
            "IA": 2,
            "IB": 2,
            "II": 3,
            "III": 3
            }
    
    # convert 9 water types to 6 new water types
    save_type = {"1": 0,
                 "3": 0,
                 "5": 1,
                 "7": 2,
                 "9": 3,
                 "I": 4,
                 "IA": 4,
                 "IB": 4,
                 "II": 5,
                 "III": 5
                 }

    water_type_label = {"1": "1, 3",
                        "3": "1, 3",
                        "5": "5",
                        "7": "7",
                        "9": "9",
                        "I": "I, IA, IB",
                        "IA": "I, IA, IB",
                        "IB": "I, IA, IB",
                        "II": "II, III",
                        "III": "II, III"
                        }
    
    # count the total number of water types
    save_type_cnts = {0: 0,
                      1: 0,
                      2: 0,
                      3: 0,
                      4: 0,
                      5: 0,
                      6: 0,
                      }
    
    for i_f, (img_f, dep_f) in tqdm(enumerate(zip(images, depths))):
        idx = int(img_f.split(".png")[0]) # get the image index
        org_img = cv2.imread(os.path.join(args.imgs_path, img_f))
        org_depth = np.load(os.path.join(args.deps_path, dep_f))
        org_img = org_img / 255.0
        org_depth = (org_depth - org_depth.min()) / (org_depth.max() - org_depth.min())
        
        for water_idx, water_type in enumerate(N_lambda.keys()):
            rand_num = rand[water_type] # how many random images should be generated for this water type
            max_depth = np.random.uniform(3, 5, size=rand_num)
            max_hori = np.random.uniform(5, 10, size=rand_num)
            # max_hori = 6.
            # B_rand = 5 - 2 * np.random.uniform(0, 1, size=rand_num)
            B_rand = np.random.uniform(0.8, 1.0, size=rand_num)
            
            for i in range(0, rand_num):
                dist = max_hori[i] * org_depth

                T_x = np.ndarray((460, 620, 3))
                T_x[:,:,0] = N_lambda[water_type][0] ** dist
                T_x[:,:,1] = N_lambda[water_type][1] ** dist
                T_x[:,:,2] = N_lambda[water_type][2] ** dist
                # T_x = (T_x-T_x.min())/(T_x.max()-T_x.min())

                B_lambda = np.ndarray((460, 620, 3))
                B_lambda[:,:,0].fill(B_rand[i]*N_lambda[water_type][0]**max_depth[i])
                B_lambda[:,:,1].fill(B_rand[i]*N_lambda[water_type][1]**max_depth[i])
                B_lambda[:,:,2].fill(B_rand[i]*N_lambda[water_type][2]**max_depth[i])

                img = org_img * T_x + B_lambda * (1 - T_x)
                # img = (img-img.min())/(img.max()-img.min())

                img_name = '{}_{}_{}.png'.format(idx, save_type[water_type], save_type_cnts[save_type[water_type]])
                
                rd = np.random.random()
                if rd < 0.1:
                    output_path = val_path
                elif rd >= 0.1 and rd < 0.2:
                    output_path = test_path
                else:
                    output_path = train_path
                    
                cv2.imwrite(os.path.join(output_path, img_name), (img * 255).astype(np.uint8))
                save_type_cnts[save_type[water_type]] += 1
    

if __name__== "__main__":

    main()