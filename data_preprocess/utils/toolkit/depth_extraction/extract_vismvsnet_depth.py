import sys
sys.path.append('.')
import os
import argparse
import argparse
import numpy as np
from utils.logging import logging
from utils.io import load_image, load_yaml

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    hparams = load_yaml(args.yaml)
    dataset_dir = os.path.abspath(args.dataset_dir)
    save_dir = os.path.join(dataset_dir, hparams.TrainDatasetSetting.depth_subdir)
    os.makedirs(save_dir, exist_ok=True)
    input_dir = os.path.join(dataset_dir, 'input')
    image_dir = os.path.join(dataset_dir, 'images')
    H = None
    W = None
    for fn in os.listdir(input_dir):
        img_new = load_image(os.path.join(image_dir, fn))
        H, W = img_new.shape[:2]
        break
    os.system(
        f'bash bash/vismvsnet_run.sh {dataset_dir} {save_dir} {W} {H} && cp -r {save_dir}/filtered/depths/*.npy {save_dir}/'
    )
    
if __name__ == '__main__':
    main()