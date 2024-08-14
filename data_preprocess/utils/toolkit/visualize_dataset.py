import sys
sys.path.append('.')
import os
import tempfile
import argparse
import numpy as np
from utils.logging import logging
from utils.toolkit.core import LiGSToolkit
from utils.visual.video import ImageCollector
from utils.io import load_image, load_yaml, load_numpy
from utils.utility import points_to_image_space, visualize_2d_points
# from submodules.gaussian_splatting.scene.colmap_loader import qvec2rotmat
from utils.util import qvec2rotmat

class DatasetVisualizer(LiGSToolkit):

    def __init__(self, dataset_dir, save_dir, alpha_subdir='masks/sam'):
        super().__init__(dataset_dir=dataset_dir, save_dir=save_dir, alpha_subdir=alpha_subdir, run_diagnosis=True)

    def run(self):
        pcd = self._init_pcd[::1]
        logging.info(f'Start visualizing dataset, point cloud size={pcd.shape}...')
        image_collector = ImageCollector(
            name='visualize_point_cloud', save_dir=self._save_dir, keep_tmp=self._keep_tmp)
        for sidx, (cam_key, cam_info) in enumerate(self._cam_infos_sorted):
            cam = self._cam_intrinsics[cam_info.camera_id]
            image_points = points_to_image_space(
                points=pcd, camera_model=cam.model, 
                R=qvec2rotmat(cam_info.qvec).T, T=cam_info.tvec,
                fx=cam.params[0], fy=cam.params[1], 
                W=cam.width, H=cam.height)
            name = os.path.splitext(cam_info.name)[0]
            alpha = load_numpy(f'{self._alphas_dir}/{name}.npy')
            alpha = np.array(alpha * 255, dtype=np.uint8)[:, :, None]
            image_collector.collect(image=visualize_2d_points(
                image_points=image_points, W=cam.width, H=cam.height, 
                flags=None, alpha=alpha, point_width=5), 
                finalize=(sidx == len(self._cam_infos_sorted) - 1))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    hparams = load_yaml(args.yaml)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    visualizer = DatasetVisualizer(
        args.dataset_dir, 
        save_dir=save_dir,
        alpha_subdir=hparams.TrainDatasetSetting.feature_settings
        .alpha.dir)
    visualizer.run()

if __name__ == '__main__':
    main()

