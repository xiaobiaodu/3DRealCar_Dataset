import sys
sys.path.append('.')
sys.path.append('submodules/gaussian_splatting')
import cv2
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from utils.logging import logging
from utils.io import save_json, load_pickle, save_pickle


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--json_fn', type=str, required=True)
    parser.add_argument('--view', type=str)
    return parser.parse_args()

def main():
    args = get_arguments()
    search_dir = args.search_dir
    save_dir=f'{search_dir}-combined/colmap_processed/pcd_rescale'
    os.makedirs(save_dir, exist_ok=True)
    track_infos = json.load(open(args.json_fn))
    views = list(args.view.split())

    sparse_save_dir = os.path.join(save_dir, 'sparse/0')
    image_save_dir = os.path.join(save_dir, 'images')
    alpha_save_dir = os.path.join(save_dir, 'masks/sam')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sparse_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(alpha_save_dir, exist_ok=True)

    dataset_names = [i for i in track_infos.keys()]
    track_ids = [track_infos[i] for i in dataset_names]

    assert len(dataset_names) == len(track_ids)

    total_pcd = []
    total_color = []
    total_normal = []
    total_extrs = {}
    total_intrs = {}
    total_crop_bbox = []
    mapping = {}
    total_idx = 1
    for view in views:
        for track_id, dataset_name in zip(track_ids, dataset_names):
            if track_id < 0:
                continue

            logging.info(f'Loading {dataset_name} track_id={track_id}')
            dataset_dir = os.path.join(search_dir, dataset_name, str(track_id), view, 'colmap_processed', 'pcd_rescale')
            images_dir = os.path.join(dataset_dir, 'images')
            alphas_dir = os.path.join(dataset_dir, 'masks/sam')
            ply_file = os.path.join(dataset_dir, "sparse/0", "points3D.ply")
            cameras_extrinsic_file = os.path.join(dataset_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(dataset_dir, "sparse/0", "cameras.bin")
            crop_bbox_file = os.path.join(dataset_dir, "sparse/0", "crop_bbox.pkl")
            if os.path.exists(ply_file) and os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
                if os.path.exists(crop_bbox_file):
                    total_crop_bbox.append(load_pickle(crop_bbox_file))
                ply_data = fetchPly(ply_file)
                total_pcd.append(ply_data.points)
                total_color.append(ply_data.colors)
                total_normal.append(ply_data.normals)
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
                for key in tqdm(cam_extrinsics.keys()):
                    image_fn = os.path.join(images_dir, cam_extrinsics[key].name)
                    alpha_fn = os.path.join(alphas_dir, cam_extrinsics[key].name)
                    save_name = f'frame_{total_idx:05d}.jpg'
                    intr = Camera(
                        id=total_idx,
                        model=cam_intrinsics[cam_extrinsics[key].camera_id].model,
                        width=cam_intrinsics[cam_extrinsics[key].camera_id].width,
                        height=cam_intrinsics[cam_extrinsics[key].camera_id].height,
                        params=cam_intrinsics[cam_extrinsics[key].camera_id].params
                    )
                    extr = Image(
                        id=total_idx,
                        qvec=cam_extrinsics[key].qvec,
                        tvec=cam_extrinsics[key].tvec,
                        camera_id=total_idx,
                        name=save_name,
                        xys=[],
                        point3D_ids=[]
                    )
                    total_extrs[total_idx] = extr
                    total_intrs[total_idx] = intr
                    os.system(f'cp {image_fn} {image_save_dir}/{save_name}')
                    os.system(f'cp {alpha_fn} {alpha_save_dir}/{save_name}')
                    total_idx += 1

    logging.info(f'total frames={len(total_intrs)}')
    write_images_binary(total_extrs, os.path.join(sparse_save_dir, 'images.bin'))
    write_cameras_binary(total_intrs, os.path.join(sparse_save_dir, 'cameras.bin'))
    total_pcd = np.concatenate(total_pcd, axis=0)
    total_color = np.concatenate(total_color, axis=0)
    total_normal = np.concatenate(total_normal, axis=0)
    logging.info(f'pcd={total_pcd.shape}, color={total_color.shape}, normal={total_normal.shape}')
    bbox = list(np.array(total_pcd.min(0)[:2], dtype=float)) + list(np.array(total_pcd.max(0)[:2], dtype=float))
    save_json(os.path.join(sparse_save_dir, 'meta.json'), {'bbox': bbox})
    storePly(os.path.join(sparse_save_dir, 'points3D.ply'), total_pcd, total_color)
    if len(total_crop_bbox) > 0:
        total_crop_bbox = np.concatenate(total_crop_bbox, axis=0)
        save_pickle(os.path.join(sparse_save_dir, 'crop_bbox.pkl'), total_crop_bbox)
        logging.info(f'Saved concatenated crop bbox into {sparse_save_dir}')
    os.system(f'ls {save_dir}/images | grep jpg > {save_dir}/trainval.meta')
    logging.info(f'Combined dataset saved into {save_dir}')

if __name__ == '__main__':
    main()


