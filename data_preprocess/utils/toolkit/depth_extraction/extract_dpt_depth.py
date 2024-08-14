import sys
sys.path.append('.')
sys.path.append('submodules/gaussian_splatting')
import os
import cv2
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from utils.logging import logging
from torchvision import transforms
from sklearn.linear_model import RANSACRegressor
from utils.io import load_colmap, save_image, load_numpy, save_numpy, load_yaml
from utils.toolkit.depth_extraction.dpt import DPTDepthModel
from utils.util import qvec2rotmat, rotmat2qvec
from utils.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def center_poses(poses, pts3d=None, enable_cam_center=False):
    
    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-10)

    if pts3d is None or enable_cam_center:
        center = poses[:, :3, 3].mean(0)
    else:
        center = pts3d.mean(0)
        
    
    up = normalize(poses[:, :3, 1].mean(0)) # (3)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    
    poses[:, :3, 3] -= center
    poses_centered = R @ poses # (N_images, 4, 4)

    if pts3d is not None:
        pts3d_centered = (pts3d - center) @ R[:3, :3].T
        # pts3d_centered = pts3d @ R[:3, :3].T - center
        return poses_centered, pts3d_centered

    return poses_centered

def rescale_depth(dataset_dir, depths, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    downscale = 1
    scale = 1    
    enable_cam_center = True

    colmap_path = os.path.join(dataset_dir, 'sparse/0')

    if colmap_path is None:
        raise ValueError(f"Cannot find colmap sparse output under {dataset_dir}, please run colmap first!")

    camdata = read_cameras_binary(os.path.join(colmap_path, 'cameras.bin'))

    # read image paths
    imdata = read_images_binary(os.path.join(colmap_path, "images.bin"))
    imkeys = np.array(sorted(imdata.keys()))

    img_names = [os.path.basename(imdata[k].name) for k in imkeys]
    img_folder = os.path.join(dataset_dir, f"images")
    img_paths = np.array([os.path.join(img_folder, name) for name in img_names])

    # only keep existing images
    exist_mask = np.array([os.path.exists(f) for f in img_paths])
    logging.info(f'{exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
    imkeys = imkeys[exist_mask]
    img_paths = img_paths[exist_mask]

    # # load masks
    # mask_folder = os.path.join(dataset_dir, alpha_subdir)

    # read intrinsics
    intrinsics = []
    for k in imkeys:
        cam = camdata[imdata[k].camera_id]
        if cam.model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
            fl_x = fl_y = cam.params[0] / downscale
            cx = cam.params[1] / downscale
            cy = cam.params[2] / downscale
        elif cam.model in ['PINHOLE', 'OPENCV']:
            fl_x = cam.params[0] / downscale
            fl_y = cam.params[1] / downscale
            cx = cam.params[2] / downscale
            cy = cam.params[3] / downscale
        else:
            raise ValueError(f"Unsupported colmap camera model: {cam.model}")
        intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
    
    intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]

    # read poses
    poses = []
    for k in imkeys:
        P = np.eye(4, dtype=np.float64)
        P[:3, :3] = imdata[k].qvec2rotmat()
        P[:3, 3] = imdata[k].tvec
        poses.append(P)
    
    poses = np.linalg.inv(np.stack(poses, axis=0)) # [N, 4, 4]

    # read sparse points
    ptsdata = read_points3D_binary(os.path.join(colmap_path, "points3D.bin"))
    ptskeys = np.array(sorted(ptsdata.keys()))
    pts3d = np.array([ptsdata[k].xyz for k in ptskeys]) # [M, 3]
    ptserr = np.array([ptsdata[k].error for k in ptskeys]) # [M]
    mean_ptserr = np.mean(ptserr)

    # center pose
    poses, pts3d = center_poses(poses, pts3d, enable_cam_center)
    logging.info(f'ColmapDataset: load poses {poses.shape}, points {pts3d.shape}')

    # rectify convention...
    poses[:, :3, 1:3] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2] *= -1

    pts3d = pts3d[:, [1, 0, 2]]
    pts3d[:, 2] *= -1

    # auto-scale
    if scale == -1:
        scale = 1 / np.linalg.norm(poses[:, :3, 3], axis=-1).min()
        logging.info(f'ColmapDataset: auto-scale {scale:.4f}')

    poses[:, :3, 3] *= scale
    pts3d *= scale

    if type != 'test':
    
        cam_near_far = [] # always extract this infomation

        logging.info(f'extracting sparse depth info...')
        # map from colmap points3d dict key to dense array index
        pts_key_to_id = np.ones(ptskeys.max() + 1, dtype=np.int64) * len(ptskeys)
        pts_key_to_id[ptskeys] = np.arange(0, len(ptskeys))

        # loop imgs
        _mean_valid_sparse_depth = 0

        for i, k in tqdm(enumerate(imkeys)):

            xys = imdata[k].xys
            xys = np.stack([xys[:, 1], xys[:, 0]], axis=-1) # invert x and y convention...
            pts = imdata[k].point3D_ids
            img_width = camdata[imdata[k].camera_id].width
            img_height = camdata[imdata[k].camera_id].height

            mask = (pts != -1) & (xys[:, 0] >= 0) & (xys[:, 0] < img_height) & (xys[:, 1] >= 0) & (xys[:, 1] < img_width)

            assert mask.any(), 'every image must contain sparse point'
            
            # valid_ids = pts_key_to_id[pts[mask]]
            valid_ids = [i for i, k in enumerate(ptskeys) if mask[i]]

            pts = pts3d[valid_ids] # points [M, 3]
            err = ptserr[valid_ids] # err [M]
            xys = xys[mask] # pixel coord [M, 2], float, original resolution!

            xys = np.round(xys / downscale).astype(np.int32) # downscale

            xys[:, 0] = xys[:, 0].clip(0, img_height - 1)
            xys[:, 1] = xys[:, 1].clip(0, img_width - 1)
            
            # calc the depth
            P = poses[i]
            depth = (P[:3, 3] - pts) @ P[:3, 2]

            # calc weight
            weight = 2 * np.exp(- (err / mean_ptserr) ** 2)

            _mean_valid_sparse_depth += depth.shape[0]

            # camera near far
            # cam_near_far.append([np.percentile(depth, 0.1), np.percentile(depth, 99.9)])
            cam_near_far.append([np.min(depth), np.max(depth)])

            # dense depth info
            imdata_name = os.path.splitext(os.path.basename(imdata[k].name))[0]
            # depth_path = os.path.join(dataset_dir, depth_subdir, imdata_name + '.npy')
            # dense_depth = load_numpy(depth_path) # [h, w]
            dense_depth = depths[imdata_name]

            # interpolate to current resolution
            dense_depth = cv2.resize(dense_depth, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

            X = dense_depth[tuple(xys.T)].reshape(-1, 1) # [M], dense
            Y = depth.reshape(-1) # [M], sparse
            W = weight.reshape(-1)

            LR = RANSACRegressor().fit(X, Y, W)
            scale = LR.estimator_.coef_[0]
            bias = LR.estimator_.intercept_

            score = np.mean((X * scale + bias - Y) ** 2)

            # must be wrong... use the most confident two samples.
            if scale < 0:
                idx_by_conf = np.argsort(W)[::-1]
                x0, y0 = X[idx_by_conf[0]][0], Y[idx_by_conf[0]]
                x1, y1 = X[idx_by_conf[1]][0], Y[idx_by_conf[1]]
                scale = (y0 - y1) / (x0 - x1)
                bias = y0 - x0 * scale
                score = np.mean((X * scale + bias - Y) ** 2)
            
                # if still wrong, use the most confident ONE sample...
                if scale < 0:
                    scale = y0 / x0
                    bias = 0
                    score = np.mean((X * scale + bias - Y) ** 2)

            logging.info(f'estimate dense depth scale by linear regression: MSE = {score:.4f}, scale = {scale:.4f}, bias = {bias:.4f}')
            dense_depth = dense_depth * scale + bias

            save_path = os.path.join(save_dir, os.path.splitext(imdata[k].name)[0] + '.npy')
            save_numpy(save_path, dense_depth)
            logging.info(f'Saved rescaled depth into {save_path}')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='resources/models/omnidata_dpt_depth_v2.ckpt')
    return parser.parse_args()

def main():
    args = get_arguments()
    hparams = load_yaml(args.yaml)
    dataset_dir = args.dataset_dir
    image_dir = os.path.join(dataset_dir, 'images')
    save_dir = os.path.join(dataset_dir, hparams.TrainDatasetSetting.feature_settings.depth.dir)
    ckpt = args.ckpt
    IMAGE_SIZE = 384

    os.makedirs(save_dir, exist_ok=True)

    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid

    logging.info(f'loading checkpoint from {ckpt}')
    checkpoint = torch.load(ckpt, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    trans_totensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    @torch.no_grad()
    def run_image(img_path):
        img = Image.open(img_path)
        W, H = img.size
        img_input = trans_totensor(img).unsqueeze(0).to(device)

        depth = model(img_input)

        depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
        depth = depth.squeeze().cpu().numpy()

        return depth

    img_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    depths = {}
    for img_path in tqdm(img_paths):
        depth = run_image(img_path)
        depths[os.path.splitext(os.path.basename(img_path))[0]] = depth
        # logging.info(f'Extracted depth for {img_path}')
    rescale_depth(dataset_dir, depths, save_dir)

if __name__ == '__main__':
    main()