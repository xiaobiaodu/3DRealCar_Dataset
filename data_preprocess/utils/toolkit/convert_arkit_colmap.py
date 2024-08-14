import sys
sys.path.append('.')
sys.path.append('submodules/gaussian_splatting')
import os
import argparse
import numpy as np
from utils.io import load_json
from utils.logging import logging
from scipy.spatial.transform import Rotation as R
from utils.utility import HomoRotX, HomoRotY, HomoRotZ
from submodules.gaussian_splatting.scene.dataset_readers import storePly
from submodules.gaussian_splatting.scene.colmap_loader import read_points3D_binary

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--camera_model', type=str, default='PINHOLE')
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    save_dir = args.save_dir
    dataset_dir = args.dataset_dir
    camera_model = args.camera_model

    images_dir = os.path.join(save_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    os.system(f'cp -r {dataset_dir}/*.jpg {images_dir}')
    
    cmd = f'colmap feature_extractor \
        --database_path {save_dir}/database.db \
        --ImageReader.camera_model {camera_model} \
        --image_path {images_dir}'
    os.system(cmd)

    images_save_dir = os.path.join(save_dir, 'create/sparse/0')
    os.makedirs(images_save_dir, exist_ok=True)
    points3D_fn = os.path.join(images_save_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')

    image_width = None
    image_height = None
    fx = None
    fy = None
    cx = None
    cy = None
    with open(os.path.join(images_save_dir, 'cameras.txt'), 'w') as fint, \
        open(os.path.join(images_save_dir, 'images.txt'), 'w') as fext:
        json_fns = [i for i in os.listdir(dataset_dir) if 'frame_' in i and os.path.splitext(i)[-1] == '.json']
        json_fns = sorted(json_fns, key=lambda x: int(os.path.splitext(x)[0].replace('frame_', '')))
        for fidx, fn in enumerate(json_fns):
            arkit_json_fn = os.path.join(dataset_dir, fn)
            name = os.path.basename(arkit_json_fn).replace('.json', '')
            if not os.path.exists(os.path.join(dataset_dir, f'{name}.jpg')):
                logging.warn(f'Skip {name} since CANNOT find corresponding image!')
                continue
            arkit_param = load_json(arkit_json_fn)
            intrinsics = arkit_param['intrinsics']
            projectionMatrix = arkit_param['projectionMatrix']
            cameraPoseARFrame = np.array(arkit_param['cameraPoseARFrame']).reshape([4, 4])

            fx = intrinsics[0]
            fy = intrinsics[4]
            cx = intrinsics[2]
            cy = intrinsics[5]
            width = int(round(2 * fx / projectionMatrix[0]))
            height = int(round(2 * fy / projectionMatrix[5]))
            if image_width is None:
                image_width = width
                image_height = height
            else:
                if image_width != width or image_height != height:
                    logging.error(f'Something wrong happend when checking intrinsics at frame {fidx}!')
                    # raise ValueError

            fint.write(f'{fidx+1} {camera_model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n')
            transform = cameraPoseARFrame.copy()
            # TODO: rotate manually, should make sure we understand why this works!
            transform = HomoRotX(np.pi) @ transform
            transform = HomoRotX(np.pi) @ np.linalg.inv(transform) # world to camera
            r = R.from_matrix(transform[:3, :3])
            rquat = r.as_quat()
            rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
            out = np.concatenate((rquat, transform[:3, 3]), axis=0)
            fext.write(f'{fidx+1} ' + ' '.join([str(a) for a in out.tolist()]) + f' {fidx+1} {name}.jpg\n\n')

    triangulated_dir = os.path.join(save_dir, 'sparse/0')
    os.makedirs(triangulated_dir, exist_ok=True)

    os.system(f'colmap exhaustive_matcher \
            --database_path {save_dir}/database.db')

    os.system(f'colmap point_triangulator \
            --database_path {save_dir}/database.db \
            --image_path {images_dir} \
            --input_path {images_save_dir} --Mapper.fix_existing_images 1 --output_path {triangulated_dir}')

    point3d_binary_fn = os.path.join(os.path.join(triangulated_dir, 'points3D.bin'))
    xyz, rgb, _ = read_points3D_binary(point3d_binary_fn)
    logging.info(f'Totally got {len(xyz)} points!')
    storePly(point3d_binary_fn.replace('.bin', '.ply'), xyz, rgb)

if __name__ == '__main__':
    main()

