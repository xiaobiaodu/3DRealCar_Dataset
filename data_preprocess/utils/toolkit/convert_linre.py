import sys
sys.path.append('.')
sys.path.append('submodules/gaussian_splatting')
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from utils.logging import logging
from scipy.spatial.transform import Rotation as R
from utils.io import save_json, save_pickle, load_image, save_image
from utils.read_write_model import Camera, Image, rotmat2qvec, write_images_binary, write_cameras_binary


def world2image(pts, intr, extr):
    pts = pts.copy()
    pts = pts @ extr.T
    pts = pts[:, :3] @ intr.T
    pts = pts / pts[:, 2:]
    return pts

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--valid_fns', type=str, required=True)
    parser.add_argument('--view', type=str, required=True)
    parser.add_argument("--crop", action='store_true')
    parser.add_argument('--scale', type=float, default=1.0)
    return parser.parse_args()

def main():
    logging.setLevel(logging.INFO)
    args = get_arguments()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    view = args.view
    valid_fns = [i.strip() for i in open(args.valid_fns).readlines()]
    ground_threshold = 0.1
    max_alpha_size = 40000
    track_info_fn = f'{dataset_dir}/track_info.txt'
    extrinsics_fn = f'{dataset_dir}/extrinsics.npy'
    intrinsics_fn = f'{dataset_dir}/intrinsics.npy'
    pcd_fn = f'{dataset_dir}/pointcloud.npz'
    image_dir = f'{dataset_dir}/images'
    # alpha_dir = f'{dataset_dir}/moving_vehicle_shadow_mask'
    alpha_dir = f'{dataset_dir}/moving_vehicle_mask'
    extrinsics = np.load(extrinsics_fn)
    intrinsics = np.load(intrinsics_fn)
    pcds = np.load(pcd_fn, allow_pickle=True)['pointcloud'].item()
    track_info = {}

    logging.info(f'Loading {track_info_fn} ...')
    for line in open(track_info_fn).readlines()[1:]:
        if len(line.strip()) <= 0:
            continue
        frame_id, track_id, object_class, alpha, box_height, box_width, box_length, box_center_x, box_center_y, box_center_z, box_rotation_w, box_rotation_x, box_rotation_y, box_rotation_z = line.strip().split()
        track_id = int(track_id)
        frame_id = int(frame_id)
        if 'Car' not in object_class:
            # logging.warn(f'skip {track_id} since {object_class}!')
            continue
        if track_id not in track_info.keys():
            track_info[track_id] = {}
        track_info[track_id][frame_id] = {
            'object_class': object_class,
            'alpha': float(alpha),
            'box_height': float(box_height),
            'box_width': float(box_width),
            'box_length': float(box_length),
            'box_center_x': float(box_center_x),
            'box_center_y': float(box_center_y),
            'box_center_z': float(box_center_z),
            'box_rotation_w': float(box_rotation_w),
            'box_rotation_x': float(box_rotation_x),
            'box_rotation_y': float(box_rotation_y),
            'box_rotation_z': float(box_rotation_z)}
    logging.info('Done')

    for selected_track_id in track_info.keys():

        logging.info(f'track_id={selected_track_id}, frames={len(track_info[selected_track_id])}')
        track_save_dir = os.path.join(save_dir, str(selected_track_id), view, 'colmap_processed/pcd_rescale')
        sparse_save_dir = os.path.join(track_save_dir, 'sparse/0')
        alpha_save_dir = os.path.join(track_save_dir, 'masks/sam')
        image_save_dir = os.path.join(track_save_dir, 'images')
        diagnosis_save_dir = os.path.join(track_save_dir, 'diagnosis')
        os.makedirs(sparse_save_dir, exist_ok=True)
        os.makedirs(alpha_save_dir, exist_ok=True)
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(diagnosis_save_dir, exist_ok=True)

        track_pts_ply = []
        track_images_bin = []
        track_cameras_bin = []
        track_bbox = []
        track_names = []
        track_image_shapes = []
        track_pts2d_bbox = []
        
        progress_bar = tqdm(range(len(track_info[selected_track_id])), desc='Processing')
        
        with open(os.path.join(track_save_dir, 'trainval.meta'), 'w') as f:
            for frame_id, param in track_info[selected_track_id].items():
                progress_bar.update(1)

                image_fn = os.path.join(image_dir, f'{frame_id:06d}_{view}.png')
                alpha_fn = os.path.join(alpha_dir, f'{frame_id:06d}_{view}.png')

                if image_fn not in valid_fns:
                    # logging.warn(f'Skip {image_fn} since Invalid!')
                    continue

                if not os.path.exists(image_fn) or not os.path.exists(alpha_fn):
                    # logging.error(f'Skip {image_fn} since Not exists!')
                    continue

                image_save_name = f'frame_{frame_id:05d}_{view}.jpg'
                image = load_image(image_fn)
                alpha = load_image(alpha_fn)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha[:, :, 0], 8)
                valid_label_idx = [i for i in range(num_labels) if stats[i][-1] > max_alpha_size]
                if len(valid_label_idx) == 2:
                    l = np.max(valid_label_idx)
                    alpha = np.array((labels == l) * 255, dtype=np.uint8)[:, :, None]
                else:
                    logging.error(f'Mismatched alpha between filter and here, please double check {alpha_fn}!')
                    continue

                H, W, _ = image.shape
                pts_ori = pcds[frame_id]
                box_center = np.array([param['box_center_x'], param['box_center_y'], param['box_center_z']])
                box_size = np.array([param['box_length'], param['box_width'], param['box_height']])
                box_rotation = np.array([param['box_rotation_w'], param['box_rotation_x'], param['box_rotation_y'], param['box_rotation_z']])
                box_rotation = R.from_quat(box_rotation[[1,2,3,0]]).as_matrix()
                box_transform = np.zeros((4, 4))
                box_transform[:3, :3] = box_rotation
                box_transform[:3, -1] = box_center
                box_transform[-1, -1] = 1
                pts_self_coord = pts_ori[:, :3] @ box_rotation
                box_center_self_coord = (box_center[None] @ box_rotation)[0]
                box_size[0] *= 1.5
                box_size[1] *= 1.5
                box_size[2] *= 1.5
                box_min = box_center_self_coord - box_size / 2
                box_max = box_center_self_coord + box_size / 2
                
                pts_valid_mask = (np.logical_and(pts_self_coord[:, :3] > box_min[None], pts_self_coord[:, :3] < box_max[None])).sum(-1) == 3
                pts = pts_ori.copy()[pts_valid_mask]
                cam_intrinsics = intrinsics[frame_id].copy()
                cam_extrinsics = extrinsics[frame_id].copy()
                pts2d_bbox = None

                if args.scale != 1.0:
                    target_H = int(round(image.shape[0] * args.scale))
                    target_W = int(round(image.shape[1] * args.scale))

                    scale_W, scale_H = target_W / alpha.shape[1], target_H / alpha.shape[0]
                    resized_alpha = cv2.resize(alpha, (target_W, target_H))[:, :, None]
                    resized_image = cv2.resize(image, (target_W, target_H))

                    cam_intrinsics[0, 0] *= scale_W
                    cam_intrinsics[0, 2] *= scale_W
                    cam_intrinsics[1, 1] *= scale_H
                    cam_intrinsics[1, 2] *= scale_H

                    alpha = resized_alpha
                    image = resized_image

                if args.crop:
                    H, W = image.shape[:2]
                    pts2d_tmp1 = np.array(world2image(pts, cam_intrinsics, np.linalg.inv(cam_extrinsics)), dtype=int)
                    pts2d_bbox = list(pts2d_tmp1[:, :2].min(0)) + list(pts2d_tmp1[:, :2].max(0))
                    border = 20
                    pts2d_bbox = np.array([
                        min(max(0, pts2d_bbox[0]-border), W-1), 
                        min(max(0, pts2d_bbox[1]-border), H-1),
                        min(max(0, pts2d_bbox[2]+border), W-1),
                        min(max(0, pts2d_bbox[3]+border), H-1)    
                    ], dtype=int)

                    if (pts2d_bbox[2] - pts2d_bbox[0]) * (pts2d_bbox[3] - pts2d_bbox[1]) <= 0:
                        continue

                    track_pts2d_bbox.append(pts2d_bbox)

                # Delete oulier points
                pts2d_tmp2 = np.array(world2image(pts, cam_intrinsics, np.linalg.inv(cam_extrinsics)), dtype=int)[:, :2]
                in_image_flag = np.logical_and(pts2d_tmp2[:, 0] >= 0, 
                            np.logical_and(pts2d_tmp2[:, 0] < alpha.shape[1],
                            np.logical_and(pts2d_tmp2[:, 1] >= 0,
                            pts2d_tmp2[:, 1] < alpha.shape[0]
                            )))
                in_alpha_flag = np.array(np.zeros(in_image_flag.shape), dtype=bool)
                for idx, (x, y) in enumerate(pts2d_tmp2[:, :2]):
                    if in_image_flag[idx] and alpha[y, x, 0] > 0:
                        in_alpha_flag[idx] = True
                pts = pts[np.logical_and(in_image_flag, in_alpha_flag)]

                if len(pts) <= 0:
                    continue

                # From world coordinates to object coordinates
                cam_extrinsics = np.linalg.inv(np.linalg.inv(box_transform) @ cam_extrinsics)
                pts = pts @ np.linalg.inv(box_transform).T
                track_pts_ply.append(pts)
                track_images_bin.append(cam_extrinsics)
                track_cameras_bin.append(cam_intrinsics)
                track_bbox.append(list(pts.min(0))[:3] + list(pts.max(0))[:3])
                track_names.append(image_save_name)

                track_image_shapes.append(list(image.shape)[:2])
                save_image(os.path.join(image_save_dir, image_save_name), image)
                save_image(os.path.join(alpha_save_dir, image_save_name), alpha)
                f.write(image_save_name + '\n')

            progress_bar.close()

            if len(track_images_bin) < 1:
                os.system(f'rm -rf {track_save_dir}')
                logging.warn(f'Deleted {track_save_dir} since NO data')
                continue

            progress_bar = tqdm(range(len(track_images_bin)), desc='Converting')
            Cameras = {}
            Images = {}
            for idx, (extr, intr, name) in enumerate(zip(track_images_bin, track_cameras_bin, track_names)):
                fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
                H, W = track_image_shapes[idx]
                Cameras[idx+1] = Camera(
                    id=idx+1, model='PINHOLE', width=W, height=H,
                    params=[fx, fy, cx, cy]
                )
                qvec = rotmat2qvec(extr[:3, :3])
                tvec = extr[:3, -1]
                Images[idx+1] = Image(
                    id=idx+1, qvec=qvec, tvec=tvec, camera_id=idx+1,
                    name=name, xys=[], point3D_ids=[]
                )
                progress_bar.update(1)

            box = np.array(track_bbox)
            box = list(box[:, :3].min(0)) + list(box[:, 3:].max(0))

            write_images_binary(Images, os.path.join(sparse_save_dir, 'images.bin'))
            write_cameras_binary(Cameras, os.path.join(sparse_save_dir, 'cameras.bin'))
            if len(track_pts2d_bbox) > 0:
                logging.warn(f'Saving track_pts2d_bbox into {sparse_save_dir}')
                save_pickle(os.path.join(sparse_save_dir, 'crop_bbox.pkl'), np.array(track_pts2d_bbox))
            save_json(os.path.join(sparse_save_dir, 'meta.json'), {'bbox': box})
            ply = np.concatenate(track_pts_ply, axis=0)[:, :3]
            wo_ground_mask = ply[:, 2] >= box[2] + ground_threshold
            ply = ply[wo_ground_mask]
            shs = np.random.random((ply.shape[0], 3)) / 255.0
            storePly(os.path.join(sparse_save_dir, 'points3D.ply'), ply, SH2RGB(shs) * 255)
            logging.info(f'Totally collected {len(ply)} points for track {selected_track_id}, totally {len(Images)} frames')
            progress_bar.close()

if __name__ == '__main__':
    main()



