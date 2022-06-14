from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np
import os
from glob import glob
import shutil
from tqdm import tqdm

# benchmark_dir = "/mnt/03_Data/UIE_Benchmark/Raw/all"
# bc_imgs = glob(os.path.join(benchmark_dir, "*.png"))
# raw_dir = "/mnt/03_Data/UIEB/data/all"
# # raw_imgs = glob(os.path.join(raw_dir, "*.png"))
# out_dir = "/mnt/03_Data/UIEB/data/test"
# os.makedirs(out_dir, exist_ok=True)
#
# for bc_f in tqdm(bc_imgs):
#     bc_img = cv2.imread(bc_f)
#     raw_imgs = glob(os.path.join(raw_dir, "*.png"))
#     for raw_f in raw_imgs:
#         raw_img = cv2.imread(raw_f)
#         # raw_img = cv2.resize(raw_img, bc_img.shape[:2])
#         if raw_img.shape[0] == bc_img.shape[0] and raw_img.shape[1] == bc_img.shape[1]:
#             ssim_score = ssim(bc_img, raw_img, multichannel=True)
#             if ssim_score >= 0.9:
#                 basename = os.path.basename(bc_f)
#                 shutil.move(raw_f, os.path.join(out_dir, basename))
#                 break

# # copy reference
# # reference_dir = "/mnt/03_Data/UIE_Benchmark/Raw/all"
# reference_dir = "/mnt/03_Data/UIEB/reference-890"
# ref_imgs = glob(os.path.join(reference_dir, "*.png"))
# test_dir = "/mnt/03_Data/UIEB/UIEB_data/test"
# test_imgs = glob(os.path.join(test_dir, "*.png"))
# out_dir = "/mnt/03_Data/UIEB/UIEB_data/test_ref"
# os.makedirs(out_dir, exist_ok=True)
#
# count = 0
# for test_f in tqdm(test_imgs):
#     test_img = cv2.imread(test_f)
#     # raw_imgs = glob(os.path.join(test_dir, "*.png"))
#     for ref_f in ref_imgs:
#         ref_img = cv2.imread(ref_f)
#         # raw_img = cv2.resize(raw_img, bc_img.shape[:2])
#         if ref_img.shape[0] == test_img.shape[0] and ref_img.shape[1] == test_img.shape[1]:
#             ssim_score = ssim(test_img, ref_img, multichannel=True)
#             if ssim_score >= 0.8:
#                 basename = os.path.basename(test_f)
#                 shutil.copy(ref_f, os.path.join(out_dir, basename))
#                 count += 1
#                 break
#
#     print("%s, %d"%(test_f, count))

# #move train ref
# train_dir = "/mnt/03_Data/UIEB/data/train"
# ref_dir = "/mnt/03_Data/UIEB/reference-890"
# out_dir = "/mnt/03_Data/UIEB/data/train_ref"
# os.makedirs(out_dir, exist_ok=True)
# train_imgs = glob(os.path.join(train_dir, "*.png"))
# for train_img in tqdm(train_imgs):
#     basename = os.path.basename(train_img)
#     shutil.copy(os.path.join(ref_dir, basename), os.path.join(out_dir, basename))


ref_dir = "/mnt/03_Data/UIEB/reference-890"
ref_all = glob(os.path.join(ref_dir, "*.png"))
train_dir = "/mnt/03_Data/UIEB/UIEB_data/train"
train_all = glob(os.path.join(train_dir, "*.png"))
basenames = set([os.path.basename(train_f) for train_f in train_all])
test_files = []
for ref in ref_all:
    ref_base = os.path.basename(ref)
    if ref_base not in basenames:
        test_files.append(ref_base)
print("test")

# os.makedirs("/mnt/03_Data/UIEB/UIEB_data/test_ref0")
# for test_f in tqdm(test_files):
#     shutil.copy(os.path.join(ref_dir, test_f), )

test_dir = "/mnt/03_Data/UIEB/UIEB_data/test"
test_imgs = glob(os.path.join(test_dir, "*.png"))
out_dir = "/mnt/03_Data/UIEB/UIEB_data/test_ref"
os.makedirs(out_dir, exist_ok=True)

count = 0
for test_f in tqdm(test_imgs):
    test_img = cv2.imread(test_f)
    # raw_imgs = glob(os.path.join(test_dir, "*.png"))
    for ref_basename in test_files:
        ref_f = os.path.join(ref_dir, ref_basename)
        ref_img = cv2.imread(ref_f)
        # raw_img = cv2.resize(raw_img, bc_img.shape[:2])
        if ref_img.shape[0] == test_img.shape[0] and ref_img.shape[1] == test_img.shape[1]:
            ssim_score = ssim(test_img, ref_img, multichannel=True)
            if ssim_score >= 0.5:
                basename = os.path.basename(test_f)
                shutil.copy(ref_f, os.path.join(out_dir, basename))
                count += 1
                break

    print("%s, %d"%(test_f, count))