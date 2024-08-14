import sys
sys.path.append('.')
import os
import numpy as np
import tempfile
import argparse
from utils.logging import logging
from utils.visual.video import ImageCollector
from utils.io import load_colmap, save_image, save_json
from utils.util import storePly
from utils.read_write_model import write_cameras_binary, write_images_binary, write_points3D_binary, read_points3D_binary, Point3D

class LiGSToolkit:

    def __init__(self, dataset_dir, save_dir, alpha_subdir='masks/sam', run_diagnosis=False):
        self._save_dir = save_dir
        self._dataset_dir = dataset_dir
        self._alpha_subdir = alpha_subdir
        self._alphas_dir = os.path.join(self._dataset_dir, self._alpha_subdir)
        self._keep_tmp = False
        self._diagnosis_dir = None
        os.makedirs(self._save_dir, exist_ok=True)
        if run_diagnosis:
            self._diagnosis_dir = os.path.join(self._save_dir, 'diagnosis')
            os.makedirs(self._diagnosis_dir, exist_ok=True)
            logging.info(f'Diagnosis directory is set to {self._diagnosis_dir}, Running in diagnosis mode!')
        self._init_pcd, self._init_color, self._init_normal, self._cam_intrinsics, self._cam_infos_sorted = \
            load_colmap(dataset_dir=dataset_dir)
    
    def process_points3D_binary(self, pcd, save_fn, inds=None):
        fn = os.path.join(self._dataset_dir, 'sparse/0/points3D.bin')
        if not os.path.exists(fn):
            logging.warn(f'Skip processing points3D.bin since NOT found!')
            return
        init_pcd_bin = read_points3D_binary(fn)
        output_pcd_bin = {}
        pcd_idx = 0
        for idx, (k, v) in enumerate(init_pcd_bin.items()):
            if inds is not None:
                if idx in inds:
                    output_pcd_bin[k] = Point3D(
                        id=v.id,
                        xyz=pcd[pcd_idx],
                        rgb=v.rgb,
                        error=v.error,
                        image_ids=v.image_ids,
                        point2D_idxs=v.point2D_idxs)
                    pcd_idx += 1
            else:
                output_pcd_bin[k] = Point3D(
                    id=v.id,
                    xyz=pcd[pcd_idx],
                    rgb=v.rgb,
                    error=v.error,
                    image_ids=v.image_ids,
                    point2D_idxs=v.point2D_idxs)
                pcd_idx += 1

        if inds is not None:
            if len(inds) != len(pcd) != len(output_pcd_bin):
                logging.error(f'Mismatch size between inds, pcd and pcd_bin, got {len(inds)} {len(pcd)} {len(output_pcd_bin)}')
                raise ValueError
        else:
            if len(pcd) != len(output_pcd_bin):
                logging.error(f'Mismatch size between pcd and pcd_bin, got {len(pcd)} {len(output_pcd_bin)}')
                raise ValueError

        write_points3D_binary(output_pcd_bin, save_fn)
        # storePly('test.ply', np.array([p.xyz for p in output_pcd_bin.values()]), np.array([p.rgb for p in output_pcd_bin.values()]))

    def process_images(self):
        save_image_dir = os.path.join(self._save_dir, 'images')
        os.makedirs(save_image_dir, exist_ok=True)
        if len(os.listdir(save_image_dir)) <= 0:
            os.system(f'ln -s {os.path.abspath(self._dataset_dir)}/images/* {os.path.abspath(save_image_dir)}/')
            logging.info(f'Link images into {save_image_dir}')
        else:
            logging.info(f'Skipped linking images since exists!')

    def process_alphas(self):
        save_alpha_dir = os.path.join(self._save_dir, self._alpha_subdir)
        os.makedirs(save_alpha_dir, exist_ok=True)
        if len(os.listdir(save_alpha_dir)) <= 0:
            os.system(f'ln -s {os.path.abspath(self._alphas_dir)}/* {os.path.abspath(save_alpha_dir)}/')
            logging.info(f'Link masks into {save_alpha_dir}')
        else:
            logging.info(f'Skipped linking alphas since exists!')
    
    def process_arkit(self):
        save_arkit_dir = os.path.join(self._save_dir, 'arkit')
        if os.path.exists(self._dataset_dir + '/arkit'):
            os.makedirs(save_arkit_dir, exist_ok=True)
            if len(os.listdir(save_arkit_dir)) <= 0:
                os.system(f'ln -s {os.path.abspath(self._dataset_dir)}/arkit/* {os.path.abspath(save_arkit_dir)}/')
                logging.info(f'Link arkit into {save_arkit_dir}')
            else:
                logging.info(f'Skipped linking arkit since exists!')
        else:
            logging.error(f'Found NO arkit directory in {self._dataset_dir}/arkit')
            raise ValueError

    def process_bbox(self, pcd, save_fn):
        bbox = list(pcd.min(0)) + list(pcd.max(0))
        bbox = list(np.array(bbox, dtype=float))
        save_json(save_fn, {'bbox': bbox})

    def save_dataset(self, pcd, colors, cam_intrinsics, cam_extrinsics, inds=None):
        if len(pcd) != len(colors):
            logging.error(f'Mismatch size between pcd and colors, got {len(pcd)} {len(colors)}')
            raise ValueError
        save_sparse_dir = os.path.join(self._save_dir, 'sparse/0')
        os.makedirs(save_sparse_dir, exist_ok=True)
        save_ply_fn = os.path.join(save_sparse_dir, 'points3D.ply')
        save_cameras_bin_fn = os.path.join(save_sparse_dir, 'cameras.bin')
        save_images_bin_fn = os.path.join(save_sparse_dir, 'images.bin')
        save_alpha_dir = os.path.join(self._save_dir, self._alpha_subdir)
        os.makedirs(save_alpha_dir, exist_ok=True)
        storePly(save_ply_fn, pcd, colors)
        logging.info(f'Write points3D.ply into {save_ply_fn}, Number of points is {len(pcd)}!')
        self.process_points3D_binary(
            pcd, 
            save_fn=os.path.join(save_sparse_dir, 'points3D.bin'),
            inds=inds)
        logging.info(f'Start writing new data into {self._save_dir} ...')
        write_cameras_binary(cam_intrinsics, save_cameras_bin_fn)
        logging.info(f'Write cameras.bin into {save_cameras_bin_fn}')
        write_images_binary(cam_extrinsics, save_images_bin_fn)
        logging.info(f'Write images.bin into {save_images_bin_fn}')
        self.process_images()
        self.process_alphas()
        self.process_arkit()
        self.process_bbox(pcd, os.path.join(save_sparse_dir, 'meta.json'))
        os.system(f'touch {self._save_dir}/.processed')

    def run(self):
        raise NotImplementedError


