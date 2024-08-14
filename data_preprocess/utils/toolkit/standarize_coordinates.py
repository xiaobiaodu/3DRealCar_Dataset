import sys
sys.path.append('.')
import os
import numpy as np
import argparse
from utils.io import load_yaml
from utils.logging import logging
from sklearn.decomposition import PCA
from utils.toolkit.core import LiGSToolkit
from utils.visual.video import ImageCollector
from scipy.spatial.transform import Rotation as R
from utils.visual.camera_param import visualize_cam_params
from utils.utility import points_to_image_space, RotX, RotY, RotZ
from utils.util import qvec2rotmat, rotmat2qvec, Image

class ForegroundCoordinateStandarizer(LiGSToolkit):

    def __init__(self, dataset_dir, save_dir, dataset_name, manual_setting,
                 epsilon=1e-2, loop_count=1, interval=0.1, 
                 interval_decay_rate=0.5, degree_decay_rate=0.5,
                 alpha_subdir='masks/sam', run_diagnosis=False):
        super().__init__(dataset_dir=dataset_dir, save_dir=save_dir, 
                         alpha_subdir=alpha_subdir, run_diagnosis=run_diagnosis)
        self._epsilon = epsilon
        self._loop_count = loop_count
        self._interval = interval
        self._interval_decay_rate = interval_decay_rate
        self._degree_decay_rate = degree_decay_rate
        self._manual = 'none'
        for line in open(manual_setting).readlines():
            if len(line.strip()) <= 0:
                continue
            splited = line.strip().split()
            if splited[0] == dataset_name:
                if len(splited) == 2:
                    self._manual = splited[1]
                    logging.warn(f'Manually rotate 180 degree along {self._manual} axis')
                    break
                else:
                    logging.warn(f'Manually rotate 180 degree along {self._manual} axis')

    def run(self):
        try:
            logging.info(f'Start find standard coordinates and corresponding transform ...')
            expect_pcd, Transform = self.standardize_coordinates(
                self._init_pcd, loop_count=self._loop_count, interval=self._interval, 
                interval_decay_rate=self._interval_decay_rate, 
                degree_decay_rate=self._degree_decay_rate, manual=self._manual)
            logging.info(f'Found transform!')
            output_pcd = (np.pad(self._init_pcd, [[0, 0], [0, 1]], constant_values=1.0) @ Transform.T)[:, :3]
            pcd_diff = np.mean(np.abs(expect_pcd - output_pcd))
            if pcd_diff > self._epsilon:
                logging.error(f'Something wrong may happen when standarizing coordinates(epsilon={pcd_diff:.2f}, expect < {self._epsilon:.2f}), please check standardize_coordinates!')
                # raise ValueError

            logging.info(f'Start adjusting camera extrinsics ...')        
            new_cam_extrinsics = {}
            for key, cam_info in self._cam_infos_sorted:
                cam = self._cam_intrinsics[cam_info.camera_id]
                r = qvec2rotmat(cam_info.qvec) @ Transform[:3, :3].T
                Rt = np.zeros((4, 4))
                Rt[:3, :3] = r.copy()
                Rt[:3, 3] = cam_info.tvec.copy()
                Rt[-1, -1] = 1.0
                C2W = np.linalg.inv(Rt)
                C2W[:3, -1] += Transform[:3, -1]
                Rt = np.linalg.inv(C2W)
                image_points = points_to_image_space(
                    points=output_pcd, camera_model=cam.model, 
                    R=Rt[:3, :3].T, T=Rt[:3, -1],
                    fx=cam.params[0], fy=cam.params[1], 
                    W=cam.width, H=cam.height)
                new_cam_extrinsics[key] = Image(
                    id=cam_info.id, 
                    qvec=rotmat2qvec(Rt[:3, :3]), 
                    tvec=Rt[:3, -1], 
                    camera_id=cam_info.camera_id, 
                    name=cam_info.name, 
                    xys=image_points[:, :2], 
                    point3D_ids=cam_info.point3D_ids,
                )
            if self._diagnosis_dir:
                visualize_cam_params(
                    points=output_pcd, 
                    cam_intrinsics=self._cam_intrinsics, 
                    cam_infos_sorted=new_cam_extrinsics.items(), 
                    save_dir=self._diagnosis_dir, 
                    name='final_pcd_standarized', 
                    keep_tmp=self._keep_tmp)
            
            self.save_dataset(
                pcd=output_pcd,
                colors=self._init_color,
                cam_intrinsics=self._cam_intrinsics,
                cam_extrinsics=new_cam_extrinsics,
                inds=None)

        except:
            raise ValueError

    def standardize_coordinates_along_axis(self, init_pcd, axis='x', interval=0.1, max_degree=90):
        axis = axis.lower()
        if axis not in ['x', 'y', 'z']:
            logging.error(f'Invalid axis name "{axis}"')
            raise ValueError
        if axis == 'x': 
            rot_func = RotX
            i = 1
            j = 2
        elif axis == 'y': 
            rot_func = RotY
            i = 0
            j = 2
        elif axis == 'z': 
            rot_func = RotZ
            i = 0
            j = 1
        min_area = np.inf
        T_out = None
        Deg_out = None
        for d in np.arange(0, max_degree, interval):
            d = d / 180 * np.pi
            T = rot_func(d)
            pcd = init_pcd @ T.T
            dist = np.max(pcd, 0) - np.min(pcd, 0)
            area = dist[i] * dist[j]
            if area < min_area:
                min_area = area
                T_out = T
                Deg_out = d
        return T_out, Deg_out, min_area

    def standardize_coordinates(self, init_pcd, loop_count=5, interval=0.1, interval_decay_rate=0.5, degree_decay_rate=0.1, manual='none'):

        Transform = np.zeros((4, 4))
        pcd = init_pcd.copy()
        rots = []
        tvecs = []
        cam_transforms = np.zeros((len(self._cam_infos_sorted), 4, 4))
        cam_transforms[:, -1, -1] = 1.0
        for k, v in self._cam_infos_sorted:
            rots.append(R.from_quat(v.qvec[[1,2,3,0]]).as_matrix())
            tvecs.append(v.tvec)
        cam_transforms[:, :3, :3] = np.array(rots)
        cam_transforms[:, :3, -1] = np.array(tvecs)
        cam_transforms = np.linalg.inv(cam_transforms)
        cam_translation = cam_transforms[:, :3, -1]

        cam_pca = PCA()
        cam_pca.fit(cam_translation)
        pca_transform = cam_pca.components_
        pca_deg = R.from_matrix(pca_transform.T).as_euler('xyz')
        
        pcd = pcd @ pca_transform.T
        final_R = pca_transform.T
        cam_trans = cam_translation @ pca_transform.T

        if np.linalg.det(pca_transform) < 0:
            logging.warn(f'flip since PCA has changed coordinates!')
            flip_transform = np.eye(3)
            flip_transform[0, 0] = -1
            pcd = pcd @ flip_transform
            final_R = final_R @ flip_transform
            cam_trans = cam_trans @ flip_transform

        cam_diff = cam_trans.max(0) - cam_trans.min(0)
        Z_axis = cam_diff.argmin()
        deg_Z_axis = np.argmax(np.abs(pca_deg - np.pi/2))
        if deg_Z_axis != Z_axis:
            logging.warn(f'Something wrong may happen, please check pcd via meshlab!')
        logging.warn(f'Z axis: use cam trans = {Z_axis}, degree = {deg_Z_axis}!')

        max_degree = 90

        for idx in range(loop_count):
            if Z_axis == 0:
                R_x, Deg_x, A_x = self.standardize_coordinates_along_axis(
                    pcd, axis='x', interval=interval, max_degree=max_degree)
                pcd = pcd @ R_x.T
                final_R = final_R @ R_x.T
            if Z_axis == 1:
                R_y, Deg_y, A_y = self.standardize_coordinates_along_axis(
                    pcd, axis='y', interval=interval, max_degree=max_degree)
                pcd = pcd @ R_y.T
                final_R = final_R @ R_y.T
            if Z_axis == 2:
                R_z, Deg_z, A_z = self.standardize_coordinates_along_axis(
                    pcd, axis='z', interval=interval, max_degree=max_degree)
                pcd = pcd @ R_z.T
                final_R = final_R @ R_z.T

        dist = np.abs(pcd.max(0) - pcd.min(0))
        Areas = np.array([dist[1] * dist[2], dist[0] * dist[2], dist[1] * dist[2]])
        X_axis = np.argmin(Areas)
        E = np.zeros((3, 3))
        E[0, X_axis] = 1
        E[2, Z_axis] = 1
        E[1, :] = -np.cross(E[0], E[2])

        if manual == 'z':
            E = E @ RotZ(np.pi) 
            logging.warn(f'Manually rotate along Z-axis 180 degree!')
        elif manual == 'x':
            E = E @ RotX(np.pi)
            logging.warn(f'Manually rotate along X-axis 180 degree!')
        elif manual == 'y':
            E = E @ RotY(np.pi)
            logging.warn(f'Manually rotate along Y-axis 180 degree!')

        final_R = final_R @ E
        pcd = pcd @ E

        # fix maually
        # d = 180-86.637
        # final_R = final_R @ RotZ(d/180*np.pi)
        # pcd = pcd @ RotZ(d/180*np.pi)

        # set origin
        T = -(pcd.min(0) + pcd.max(0)) / 2
        pcd = pcd + T
        
        Transform[:3, :3] = final_R.T
        Transform[:3, -1] = T
        Transform[-1, -1] = 1

        return pcd, Transform

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--manual_setting', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.addFileHandler(os.path.join(args.save_dir, 'pcd_standard.log'))
    hparams = load_yaml(args.yaml)
    extrator = ForegroundCoordinateStandarizer(
        dataset_dir=args.dataset_dir, 
        dataset_name=args.dataset_name,
        save_dir=args.save_dir,
        epsilon=hparams.TrainDatasetSetting.pcd_standard_epsilon,
        loop_count=hparams.TrainDatasetSetting.pcd_standard_loop_count,
        interval=hparams.TrainDatasetSetting.pcd_standard_interval,
        interval_decay_rate=hparams.TrainDatasetSetting.pcd_standard_interval_decay_rate,
        degree_decay_rate=hparams.TrainDatasetSetting.pcd_standard_degree_decay_rate,
        manual_setting=args.manual_setting,
        alpha_subdir=hparams.TrainDatasetSetting.feature_settings.alpha.dir, 
        run_diagnosis=hparams.TrainDatasetSetting.run_diagnosis)
    extrator.run()

if __name__ == '__main__':
    main()





