import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils.utility import *
from matplotlib import pyplot as plt
from utils.io import load_image, save_image

def save_img(img):
    x = img.permute([1,2,0]).cpu().detach().numpy()
    save_image('test.png', x * 255)

def extract_image_by_freq(image, cutoff_freq):
    # image: C x H x W
    if cutoff_freq <= 0:
        return image
    _, H, W = image.shape
    kernel = get_filter2d(H, W, cutoff_freq)
    fft = fourier2d(image)
    inv_fft = inv_fourier2d(fft * kernel.to(image.device))
    return inv_fft

def calculate_freq_diff(name, real, pred, mask, cutoff_freq=1, diagnosis_dir=None):
    if diagnosis_dir is not None:
        os.makedirs(diagnosis_dir, exist_ok=True)
        l1_dir = os.path.join(diagnosis_dir, 'l1')
        psnr_dir = os.path.join(diagnosis_dir, 'psnr')
        ssim_dir = os.path.join(diagnosis_dir, 'ssim')
        real_dir = os.path.join(diagnosis_dir, 'real')
        pred_dir = os.path.join(diagnosis_dir, 'pred')
        os.makedirs(l1_dir, exist_ok=True)
        os.makedirs(psnr_dir, exist_ok=True)
        os.makedirs(ssim_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
    real = real * mask
    real_ = extract_image_by_freq(real, cutoff_freq=cutoff_freq)
    pred_ = extract_image_by_freq(pred, cutoff_freq=cutoff_freq)
    l1, l1_img = masked_l1(real_, pred_, mask)
    psnr, psnr_img = masked_psnr(real_, pred_, mask)
    ssim, ssim_img = masked_ssim(real_, pred_, mask)
    if diagnosis_dir is not None:
        l1_img = (l1_img - l1_img.min()) / (l1_img.max() - l1_img.min())
        l1_img = l1_img.permute([1, 2, 0]).detach().cpu().numpy() * 255
        save_image(os.path.join(l1_dir, f'{freq}_{name}'), l1_img)
        psnr_img = psnr_img.permute([1, 2, 0]).detach().cpu().numpy() * 255
        save_image(os.path.join(psnr_dir, f'{freq}_{name}'), psnr_img)
        ssim_img = ssim_img.permute([1, 2, 0]).detach().cpu().numpy() * 255
        save_image(os.path.join(ssim_dir, f'{freq}_{name}'), ssim_img)
        save_image(os.path.join(real_dir, f'{freq}_{name}'), real_.permute([1, 2, 0]).detach().cpu().numpy() * 255)
        save_image(os.path.join(pred_dir, f'{freq}_{name}'), pred_.permute([1, 2, 0]).detach().cpu().numpy() * 255)
    return l1, psnr, 1 - ssim

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--diagnosis_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--test_name', type=str)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--mode', type=str, choices=['v1', 'v2'], required=True, default='v1')
    parser.add_argument("--run_unpaired", action='store_true')
    parser.add_argument("--run_tests", action='store_true')
    parser.add_argument("--save_log", action='store_true')
    return parser.parse_args()
    
def main():
    args = get_arguments()
    diagnosis_dir = args.diagnosis_dir
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    f = plt.figure()
    ax = f.add_subplot(111)
    total_l1s = []
    total_psnrs = []
    total_ssims = []
    for name in os.listdir(base_dir+'/real'):
        real_fn = os.path.join(base_dir, 'real', name)
        pred_fn = os.path.join(base_dir, 'pred', name)
        mask_fn = os.path.join(base_dir, 'mask', name)
        real = torch.Tensor(load_image(real_fn)).cuda().permute([2, 0, 1]) / 255
        pred = torch.Tensor(load_image(pred_fn)).cuda().permute([2, 0, 1]) / 255
        mask = torch.Tensor(load_image(mask_fn)).cuda().permute([2, 0, 1])[0,None]
        _, H, W = real.shape
        freqs = list(np.arange(1,min(H, W)//2)) + [0]
        mask = (mask > 240).float()
        real = real * mask
        l1s = []
        psnrs = []
        ssims = []
        for freq in tqdm(freqs):
            l1, psnr, ssim = calculate_freq_diff(name, real, pred, mask, cutoff_freq=freq, diagnosis_dir=diagnosis_dir)
            l1s.append(l1.detach().cpu().numpy())
            psnrs.append(psnr.detach().cpu().numpy())
            ssims.append(ssim.detach().cpu().numpy())
        l1s = np.array(l1s)
        l1s = (l1s - l1s.min()) / (l1s.max() - l1s.min())
        psnrs = np.array(psnrs)
        psnrs = (psnrs - psnrs.min()) / (psnrs.max() - psnrs.min())
        ssims = np.array(ssims)
        ssims = (ssims - ssims.min()) / (ssims.max() - ssims.min())
        total_l1s.append(l1s)
        total_psnrs.append(psnrs)
        total_ssims.append(ssims)

    xlabels = list(np.arange(len(freqs)))
    ax.fill_between(xlabels, np.min(total_l1s, 0), np.max(total_l1s, 0), color='r', alpha=0.5)
    ax.fill_between(xlabels, np.min(total_psnrs, 0), np.max(total_psnrs, 0), color='g', alpha=0.5)
    ax.fill_between(xlabels, np.min(total_ssims, 0), np.max(total_ssims, 0), color='b', alpha=0.5)
    ax.legend(['l1', 'psnr', 'ssim'])
    ax.set_xticks(freqs)
    ax.set_ylabel('rescaled value')
    ax.set_xlabel('accumlated frequency')
    plt.savefig(f'{diagnosis_dir}/{dataset_name}.png')

if __name__ == '__main__':
    main()
