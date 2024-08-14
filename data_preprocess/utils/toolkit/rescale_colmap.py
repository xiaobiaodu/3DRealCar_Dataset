import sys
sys.path.append('.')
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.logging import logging
from sklearn.decomposition import PCA
from utils.toolkit.core import LiGSToolkit
from utils.visual.video import ImageCollector
from scipy.spatial.transform import Rotation as R
from utils.io import load_yaml, load_json, load_image, load_numpy
from utils.visual.camera_param import visualize_cam_params
from utils.utility import points_to_image_space, \
    visualize_2d_points, check_points_in_image, HomoRotX, HomoRotY, HomoRotZ
from utils.util import qvec2rotmat, Image

class RescalePCDExtractor(LiGSToolkit):
    """
    rescaling pcd and modify extrinsics, mainly used in colmap
    """

    def __init__(self, dataset_dir, save_dir, scales=[1,1,1], alpha_subdir='masks/sam', run_diagnosis=False):
        super().__init__(dataset_dir=dataset_dir, save_dir=save_dir, alpha_subdir=alpha_subdir, run_diagnosis=run_diagnosis)
        self._scales_default = np.array(scales)
        # self._scales = self._scales_default
        self._scales = self.get_arkit_cameras()

    def get_arkit_cameras(self):
        arkit_dir = os.path.join(self._dataset_dir, 'arkit')
        used_count = 0
        if os.path.exists(arkit_dir):
            arkit_cam_trans = []
            colmap_cam_trans = []
            for k, v in self._cam_infos_sorted:
                arkit_fn = os.path.join(arkit_dir, os.path.splitext(v.name)[0] + '.json')
                if not os.path.exists(arkit_fn):
                    logging.warn(f'Skip {arkit_fn} since corresponding arkit file NOT found!')
                    continue
                used_count += 1
                arkit_param = load_json(arkit_fn)
                cameraPoseARFrame = np.array(arkit_param['cameraPoseARFrame']).reshape([4, 4])
                cameraPoseARFrame = HomoRotX(np.pi) @ cameraPoseARFrame
                qvec = v.qvec[[1, 2, 3, 0]].copy()
                tvec = v.tvec.copy()
                Rt = np.zeros((4,4))
                Rt[:3, :3] = R.from_quat(qvec).as_matrix()
                Rt[:3, -1] = tvec
                Rt[-1, -1] = 1
                Rt = np.linalg.inv(Rt)
                arkit_cam_trans.append(cameraPoseARFrame[:3, -1])
                colmap_cam_trans.append(Rt[:3, -1])
            arkit_cam_trans = np.array(arkit_cam_trans)
            colmap_cam_trans = np.array(colmap_cam_trans)

            # colmap trans are standard
            colmap_pca = PCA()
            colmap_pca.fit(colmap_cam_trans)
            arkit_pca = PCA()
            arkit_pca.fit(arkit_cam_trans)
            arkit_pca_T = arkit_pca.components_.T
            if np.linalg.det(arkit_pca_T) < 0:
                flip_transform = np.eye(3)
                flip_transform[0, 0] = -1
                arkit_pca_T = arkit_pca_T @ flip_transform
            colmap_pca_T = colmap_pca.components_
            if np.linalg.det(colmap_pca_T) < 0:
                flip_transform = np.eye(3)
                flip_transform[0, 0] = -1
                colmap_pca_T = colmap_pca_T @ flip_transform

            arkit_cam_trans = arkit_cam_trans @ arkit_pca_T
            arkit_cam_trans = arkit_cam_trans - arkit_cam_trans.mean(0) + colmap_cam_trans.mean(0)
            arkit_cam_trans = arkit_cam_trans @ colmap_pca_T
            scales = (arkit_cam_trans.max(0) - arkit_cam_trans.min(0)) / (colmap_cam_trans.max(0) - colmap_cam_trans.min(0))
            scales = np.ones(3) * scales[:2].mean()

            logging.warn(f'Automatically set rescales to {scales}!')
            
            pca_save_fn=os.path.join(self._save_dir, 'pca.png')
            f = plt.figure()
            ax = f.add_subplot(projection='3d')
            ax.scatter(colmap_cam_trans[:, 0], colmap_cam_trans[:, 1], colmap_cam_trans[:, 2])
            ax.scatter(arkit_cam_trans[:, 0], arkit_cam_trans[:, 1], arkit_cam_trans[:, 2])
            f.savefig(pca_save_fn)
            logging.info(f'Rescaling camera trace saved into {pca_save_fn}')

            if scales.max() > 2 or scales.min() < 0.8:
                logging.warn(f'Invalid scales: {scales}, may rerun colmap!')
                raise ValueError

            return scales
        else:
            logging.warn(f'ARKit directory {arkit_dir} NOT found, use {self._scales_default} instead!')
            return self._scales_default
        logging.info(f'Only {used_count}/{self._cam_infos_sorted}[{used_count/self._cam_infos_sorted*100:.2f}%] data is used for rescaling!')

    def run(self):
        try:
            self._init_pcd = self._init_pcd * self._scales
            bbox_length = list(self._init_pcd.max(0) - self._init_pcd.min(0))
            logging.warn(f'final pcd bbox: {bbox_length}')
            new_cam_extrinsics = {}
            image_collector = None
            if self._diagnosis_dir:
                image_collector = ImageCollector(
                    name='final_xys', save_dir=self._diagnosis_dir, keep_tmp=self._keep_tmp)
            for fidx, (cam_key, cam_info) in enumerate(self._cam_infos_sorted):
                cam = self._cam_intrinsics[cam_info.camera_id]
                # TODO: these are fake xys and point3D_ids
                image_points = points_to_image_space(
                    points=self._init_pcd, camera_model=cam.model, 
                    R=qvec2rotmat(cam_info.qvec).T, T=cam_info.tvec * self._scales,
                    fx=cam.params[0], fy=cam.params[1], 
                    W=cam.width, H=cam.height)
                point3D_ids = []
                for idx, point in enumerate(image_points[:, :2]):
                    if check_points_in_image(point, cam.width, cam.height):
                        point3D_ids.append(idx)
                    else:
                        point3D_ids.append(-1)
                new_cam_info = Image(
                    id=cam_info.id, 
                    qvec=cam_info.qvec, 
                    tvec=cam_info.tvec * self._scales, 
                    camera_id=cam_info.camera_id, 
                    name=cam_info.name, 
                    xys=image_points[:, :2], 
                    point3D_ids=point3D_ids)
                new_cam_extrinsics[cam_key] = new_cam_info
                alpha = load_numpy(os.path.join(self._alphas_dir, os.path.splitext(cam_info.name)[0] + '.npy'))
                alpha = np.array(alpha * 255, dtype=np.uint8)[:, :, None]
                # use bbox to find out invalid alphas
                if self._diagnosis_dir:
                    image_collector.collect(image=visualize_2d_points(
                        image_points=new_cam_info.xys, W=cam.width, H=cam.height, 
                        flags=None, alpha=alpha, point_width=5), 
                        finalize=(fidx == len(self._cam_infos_sorted) - 1))
            if self._diagnosis_dir:
                visualize_cam_params(
                    points=self._init_pcd, 
                    cam_intrinsics=self._cam_intrinsics, 
                    cam_infos_sorted=new_cam_extrinsics.items(), 
                    save_dir=self._diagnosis_dir, 
                    name='final_pcd_clean', 
                    keep_tmp=self._keep_tmp)
            
            self.save_dataset(
                pcd=self._init_pcd,
                colors=self._init_color,
                cam_intrinsics=self._cam_intrinsics,
                cam_extrinsics=new_cam_extrinsics,
                inds=None)

        except:
            raise ValueError

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.addFileHandler(os.path.join(args.save_dir, 'pcd_rescale.log'))
    hparams = load_yaml(args.yaml)
    extrator = RescalePCDExtractor(
        args.dataset_dir, 
        save_dir=args.save_dir,
        scales=hparams.TrainDatasetSetting.rescales,
        alpha_subdir=hparams.TrainDatasetSetting.feature_settings.alpha.dir, 
        run_diagnosis=hparams.TrainDatasetSetting.run_diagnosis)
    extrator.run()

if __name__ == '__main__':
    main()





