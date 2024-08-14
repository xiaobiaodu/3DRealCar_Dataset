import sys
sys.path.append('.')
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from utils.logging import logging
from utils.toolkit.core import LiGSToolkit
from utils.io import load_pickle, load_yaml, load_image, load_numpy
from utils.visual.video import ImageCollector
from utils.visual.camera_param import visualize_cam_params
# from submodules.gaussian_splatting.scene.dataset_readers import storePly
# from submodules.gaussian_splatting.scene.colmap_loader import qvec2rotmat, Image
from utils.utility import points_to_image_space, visualize_2d_points, check_points_in_image
from utils.util import Image, qvec2rotmat

class ForegroundPCDExtractor(LiGSToolkit):

    def __init__(self, dataset_dir, save_dir, pixel_border=5, 
                 check_bbox_valid_length=20, check_bbox_valid_border=20, 
                 check_bbox_valid_bbox_diff=50, alpha_border=1,
                 alpha_subdir='masks/sam', run_diagnosis=False, alpha_threshold=0):
        super().__init__(dataset_dir=dataset_dir, save_dir=save_dir, alpha_subdir=alpha_subdir, run_diagnosis=run_diagnosis)
        self._pixel_border = pixel_border
        self._complete_object_names = []
        self._selected_inds = np.arange(len(self._init_pcd))
        self._check_bbox_valid_length = check_bbox_valid_length
        self._check_bbox_valid_border = check_bbox_valid_border
        self._check_bbox_valid_bbox_diff = check_bbox_valid_bbox_diff
        self._alpha_border = alpha_border
        self._alpha_threshold = alpha_threshold
        logging.info(f'check_bbox_valid_length={check_bbox_valid_length}')
        logging.info(f'check_bbox_valid_border={check_bbox_valid_border}')
        logging.info(f'check_bbox_valid_bbox_diff={check_bbox_valid_bbox_diff}')
        logging.info(f'alpha_border={alpha_border}')
        logging.info(f'alpha_threshold={alpha_threshold}')

    def run(self):
        try:
            logging.info(f'Number of points(Initial) is {len(self._init_pcd)}')
            pcd, colors, normals = self.remove_points_outside_image(
                init_pcd=self._init_pcd, init_colors=self._init_color, init_normals=self._init_normal, keep_tmp=self._keep_tmp)
            logging.info(f'Number of points(remove outside image) is {len(pcd)}')
            if self._diagnosis_dir:
                visualize_cam_params(
                    pcd, 
                    cam_intrinsics=self._cam_intrinsics, 
                    cam_infos_sorted=self._cam_infos_sorted, 
                    save_dir=self._diagnosis_dir, 
                    name='remove_outside_image', 
                    keep_tmp=self._keep_tmp)
            pcd, colors, normals = self.remove_points_outside_alpha(
                init_pcd=pcd, init_colors=colors, init_normals=normals, border=self._alpha_border, keep_tmp=self._keep_tmp)
            logging.info(f'Number of points(remove outside alpha) is {len(pcd)}')
            if self._diagnosis_dir:
                visualize_cam_params(
                    pcd, 
                    cam_intrinsics=self._cam_intrinsics, 
                    cam_infos_sorted=self._cam_infos_sorted, 
                    save_dir=self._diagnosis_dir, 
                    name='remove_outside_alpha', 
                    keep_tmp=self._keep_tmp)

            # Exporting new bins and ply
            final_valid_names = []
            new_cam_extrinsics = {}
            image_collector = None
            checker_image_collector = None
            if self._diagnosis_dir:
                image_collector = ImageCollector(
                    name='final_xys', save_dir=self._diagnosis_dir, keep_tmp=self._keep_tmp)
                checker_image_collector = ImageCollector(
                    name='final_alpha_checker', save_dir=self._diagnosis_dir, keep_tmp=self._keep_tmp)
            for fidx, (cam_key, cam_info) in enumerate(self._cam_infos_sorted):
                cam = self._cam_intrinsics[cam_info.camera_id]
                # TODO: these are fake xys and point3D_ids
                image_points = points_to_image_space(
                    points=pcd, camera_model=cam.model, 
                    R=qvec2rotmat(cam_info.qvec).T, T=cam_info.tvec,
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
                    tvec=cam_info.tvec, 
                    camera_id=cam_info.camera_id, 
                    name=cam_info.name, 
                    xys=image_points[:, :2], 
                    point3D_ids=point3D_ids)
                new_cam_extrinsics[cam_key] = new_cam_info
                name = os.path.splitext(cam_info.name)[0]
                alpha = load_numpy(os.path.join(self._alphas_dir, name + '.npy'))
                alpha = np.array(alpha * 255, dtype=np.uint8)[:, :, None]
                meta_fn = os.path.join(self._alphas_dir, name + '.pkl')
                if not os.path.exists(meta_fn):
                    continue
                alpha_bbox = self.load_alpha_meta_bbox(meta_fn, W=cam.width, H=cam.height)
                # use bbox to find out invalid alphas
                annotated_alpha, valid_regions = self.check_bbox_diff_between_points_and_alpha(
                    image_points=image_points, alpha=alpha.copy(), alpha_bbox=alpha_bbox)
                if len(valid_regions) != 1:
                    logging.warn(f'Should be only 1 valid regions but got {len(valid_regions)} in {cam_info.name}, will be skipped, more details: {valid_regions}')
                else:
                    final_valid_names.append(cam_info.name)
                if self._diagnosis_dir:
                    image_collector.collect(image=visualize_2d_points(
                        image_points=new_cam_info.xys, W=cam.width, H=cam.height, 
                        flags=None, alpha=alpha, point_width=5), 
                        finalize=(fidx == len(self._cam_infos_sorted) - 1))
                    checker_image_collector.collect(
                        image=annotated_alpha, 
                        finalize=(fidx == len(self._cam_infos_sorted) - 1))
            if self._diagnosis_dir:
                visualize_cam_params(
                    points=pcd, 
                    cam_intrinsics=self._cam_intrinsics, 
                    cam_infos_sorted=new_cam_extrinsics.items(), 
                    save_dir=self._diagnosis_dir, 
                    name='final_pcd_clean', 
                    keep_tmp=self._keep_tmp)

            with open(os.path.join(self._save_dir, 'trainval.meta'), 'w') as f:
                for name in final_valid_names:
                    f.write(name+'\n')
            logging.info(f'Valid names are: ')
            logging.info(' '.join(final_valid_names))
            logging.info(f'Totally found {len(final_valid_names)}/{len(self._cam_infos_sorted)} ({len(final_valid_names) / len(self._cam_infos_sorted) * 100:.2f}%) valid frames!')

            self.save_dataset(
                pcd=pcd,
                colors=colors,
                cam_intrinsics=self._cam_intrinsics,
                cam_extrinsics=new_cam_extrinsics,
                inds=self._selected_inds)
            
        except Exception as e:
            logging.error(f'extract foreground failed since {e}')
            raise ValueError

    def remove_points_outside_image(self, init_pcd, init_colors, init_normals, keep_tmp=False):
        # using image-space pixels of point cloud to find foreground points
        pcd = init_pcd.copy()
        colors = init_colors.copy()
        normals = init_normals.copy()
        image_collector = None
        checker_image_collector = None
        if self._diagnosis_dir:
            image_collector = ImageCollector(
                name='remove_outside_image_wip', save_dir=self._diagnosis_dir, keep_tmp=self._keep_tmp)
            checker_image_collector = ImageCollector(
                name='alpha_checker_wip', save_dir=self._diagnosis_dir, keep_tmp=self._keep_tmp)
        progress_bar = tqdm(range(len(self._cam_infos_sorted)), desc='Remove outside image')
        for sidx, (cam_key, cam_info) in enumerate(self._cam_infos_sorted):
            cam = self._cam_intrinsics[cam_info.camera_id]
            meta_fn = os.path.join(self._alphas_dir, os.path.splitext(cam_info.name)[0] + '.pkl')
            if not os.path.exists(meta_fn):
                continue
            xmin, ymin, xmax, ymax = self.load_alpha_meta_bbox(meta_fn, W=cam.width, H=cam.height)
            if not check_points_in_image(point=[xmin, ymin], W=cam.width, H=cam.height, border=self._pixel_border) or \
                not check_points_in_image(point=[xmax, ymax], W=cam.width, H=cam.height, border=self._pixel_border):
                logging.warn(f'file "{cam_info.name}" contains incomplete object, skipped and will not be used in the next stage!')
                continue
            image_points = points_to_image_space(
                points=pcd, camera_model=cam.model, 
                R=qvec2rotmat(cam_info.qvec).T, T=cam_info.tvec,
                fx=cam.params[0], fy=cam.params[1], 
                W=cam.width, H=cam.height)
            alpha = load_numpy(os.path.join(self._alphas_dir, os.path.splitext(cam_info.name)[0] + '.npy'))
            alpha = np.array(alpha * 255, dtype=np.uint8)[:, :, None]
            annotated_alpha, valid_regions = self.check_bbox_diff_between_points_and_alpha(
                image_points, alpha=alpha.copy(), 
                alpha_bbox=[xmin, ymin, xmax, ymax])
            if len(valid_regions) != 1:
                logging.warn(f'Should be only 1 valid regions but got {len(valid_regions)} in {cam_info.name}, will be skipped, more details: {valid_regions}')
                continue
            self._complete_object_names.append(cam_info.name)
            flags = (image_points[:, 0] >= 0) & \
                    (image_points[:, 0] < cam.width) & \
                    (image_points[:, 1] >= 0) & \
                    (image_points[:, 1] < cam.height)
            if self._diagnosis_dir:
                image_collector.collect(image=visualize_2d_points(
                    image_points=image_points, W=cam.width, H=cam.height, 
                    flags=flags, alpha=None, point_width=5), 
                    finalize=False)
                checker_image_collector.collect(
                    image=annotated_alpha, finalize=False)
            pcd = pcd[flags]
            if len(pcd) <= 0:
                logging.error(f'ALL points are removed, something wrong may happen!')
                raise ValueError
            colors = colors[flags]
            normals = normals[flags]
            self._selected_inds = self._selected_inds[flags]
            progress_bar.update(1)
        progress_bar.close()
        # Some frames were skipped, so have to finalize manually!
        if self._diagnosis_dir:
            image_collector.finalize()
            checker_image_collector.finalize()
        logging.warn(f'Only {len(self._complete_object_names)}/{len(self._cam_infos_sorted)} [{len(self._complete_object_names)/len(self._cam_infos_sorted)*100:.2f}%] images are used to clean PCD!')
        return pcd, colors, normals

    def remove_points_outside_alpha(self, init_pcd, init_colors, init_normals, border=1, keep_tmp=False):
        if len(self._complete_object_names) <= 0:
            logging.error(f'NO valid files were detected, please check remove_points_outside_image for more details!')
            raise ValueError
        # using alpha pixels of point cloud to find foreground points
        pcd = init_pcd.copy()
        colors = init_colors.copy()
        normals = init_normals.copy()
        image_collector = None
        if self._diagnosis_dir:
            image_collector = ImageCollector(
                name='remove_outside_alpha_wip', save_dir=self._diagnosis_dir, keep_tmp=self._keep_tmp)
        progress_bar = tqdm(range(len(self._cam_infos_sorted)), desc='Remove outside alpha')
        for sidx, (cam_key, cam_info) in enumerate(self._cam_infos_sorted):
            if self._complete_object_names is not None:
                if cam_info.name not in self._complete_object_names:
                    continue
            alpha = load_numpy(os.path.join(self._alphas_dir, os.path.splitext(cam_info.name)[0] + '.npy'))
            alpha = np.array(alpha * 255, dtype=np.uint8)[:, :, None]
            cam = self._cam_intrinsics[cam_info.camera_id]
            image_points = points_to_image_space(
                points=pcd, camera_model=cam.model, 
                R=qvec2rotmat(cam_info.qvec).T, T=cam_info.tvec,
                fx=cam.params[0], fy=cam.params[1], 
                W=cam.width, H=cam.height)
            flags = np.array([True] * len(image_points))
            for idx, (x, y) in enumerate(image_points[:, :2]):
                if check_points_in_image([x, y], cam.width, cam.height, border=0):
                    x = int(np.round(x))
                    y = int(np.round(y))
                    l = max(0, x - border)
                    r = min(x + border, cam.width)
                    t = max(0, y - border)
                    b = min(y + border, cam.height)
                    flags[idx] &= (alpha[t:b,l:r] == 255).any()
                else:
                    flags[idx] = False
            if self._diagnosis_dir:
                image_collector.collect(image=visualize_2d_points(
                    image_points=image_points, W=cam.width, H=cam.height, 
                    flags=flags, alpha=alpha, point_width=5), 
                    finalize=False)
            pcd = pcd[flags]
            if len(pcd) <= 0:
                logging.error(f'ALL points are removed, something wrong may happen!')
                raise ValueError
            colors = colors[flags]
            normals = normals[flags]
            self._selected_inds = self._selected_inds[flags]
            progress_bar.update(1)
        progress_bar.close()
        # Some frames were skipped, so have to finalize manually!
        if self._diagnosis_dir:
            image_collector.finalize()
        return pcd, colors, normals
    
    def check_bbox_diff_between_points_and_alpha(self, image_points, alpha, alpha_bbox):
        H, W = alpha.shape[:2]
        alpha = np.array(alpha, dtype=np.uint8)
        # Check projected bbox and connected components' bbox to remove invalid frames
        xmin, ymin, xmax, ymax = alpha_bbox
        proj_xmin = 1e10
        proj_xmax = -1
        proj_ymin = 1e10
        proj_ymax = -1
        for point in image_points[:, :2]:
            if check_points_in_image(point=point, W=W, H=H):
                x, y = point
                if x < proj_xmin:
                    proj_xmin = x
                if x > proj_xmax:
                    proj_xmax = x
                if y < proj_ymin:
                    proj_ymin = y
                if y > proj_ymax:
                    proj_ymax = y
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha, 8)
        valid_regions = []
        cv2.rectangle(alpha, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
        for x, y, w, h, a in stats:
            if w  > self._check_bbox_valid_length and \
                h > self._check_bbox_valid_length and \
                    np.abs(w - W) > self._check_bbox_valid_border and \
                        np.abs(h - H) > self._check_bbox_valid_border:
                diff = (np.abs(x - xmin) + np.abs(y - ymin) + np.abs(x + w - xmax) + np.abs(y + h - ymax)) / 4
                if diff < self._check_bbox_valid_bbox_diff:
                    cv2.rectangle(alpha, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    valid_regions.append([x, y, x+w, y+h, diff])
                else:
                    cv2.rectangle(alpha, (x, y), (x+w, y+h), (0, 0, 255), 3)
        return alpha, valid_regions

    def load_alpha_meta_bbox(self, meta_fn, W, H):
        if not os.path.exists(meta_fn):
            logging.error(f'meta file "{meta_fn}" NOT exists, something wrong may happened in [segmentation] stage, please check!')
            raise FileNotFoundError
        meta = load_pickle(meta_fn)
        meta_image_shape = meta['image_shape']
        if W != meta_image_shape[1] or H != meta_image_shape[0]:
            logging.error(f'mismatched image shape between came info and alpha meta, got [{H}, {W}] and {meta_image_shape} respectively!')
            raise ValueError
        return np.array(meta['boxes_xyxy'][0])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.addFileHandler(os.path.join(args.save_dir, 'pcd_clean.log'))
    hparams = load_yaml(args.yaml)
    extrator = ForegroundPCDExtractor(
        args.dataset_dir, 
        save_dir=args.save_dir,
        alpha_subdir=hparams.TrainDatasetSetting.feature_settings.alpha.dir,
        check_bbox_valid_length=hparams.TrainDatasetSetting.check_bbox_valid_length,
        check_bbox_valid_border=hparams.TrainDatasetSetting.check_bbox_valid_border,
        check_bbox_valid_bbox_diff=hparams.TrainDatasetSetting.check_bbox_valid_bbox_diff,
        alpha_border=hparams.TrainDatasetSetting.alpha_border,
        pixel_border=hparams.TrainDatasetSetting.pcd_clean_pixel_border,
        alpha_threshold=hparams.TrainDatasetSetting.alpha_threshold,
        run_diagnosis=hparams.TrainDatasetSetting.run_diagnosis)
    extrator.run()

if __name__ == '__main__':
    main()





