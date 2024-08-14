import sys
sys.path.append('submodules/gaussian_splatting')
import os
import numpy as np
from tqdm import tqdm
from utils.logging import logging
from plyfile import PlyData, PlyElement
from utils.utility import RotX, RotY, RotZ
from utils.io import load_image, load_json, load_obj
from utils.util import read_points3D_binary, storePly

class _AbstractDatasetAdaptor(object):
    """
    Abstract class for DatasetAdaptor, used for formatting directories and files for dataset collected through various tools. The final directory format looks like what we eventually get after reconstructing via COLMAP.
    
    DATASET_NAME
    ├── images
    └── sparse
        └── 0
            ├── cameras.bin
            ├── images.bin
            ├── points3D.ply (optional, use random if NOT found, please check logging)
    """

    def __init__(self, search_dir, save_dir, valid_image_suffix=['.jpg', '.png']):
        self._valid_image_suffix = valid_image_suffix
        self._name = os.path.basename(search_dir)
        self._search_dir = search_dir
        self._save_dir = save_dir
        self._sparse_save_dir = f'{self._save_dir}/sparse/0'
        self._images_bin_fn = os.path.join(self._sparse_save_dir, 'images.bin')
        self._cameras_bin_fn = os.path.join(self._sparse_save_dir, 'cameras.bin')
        self._images_dir = os.path.join(self._save_dir, 'images')
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._sparse_save_dir, exist_ok=True)
        os.makedirs(self._images_dir, exist_ok=True)

    def __call__(self):
        self.check_unprocessed()
        self._internal_call()
        self.check_processed()
        os.system(f'touch {self._save_dir}/.dataset')

    def _internal_call(self):
        raise NotImplementedError

    def check_unprocessed(self):
        raise NotImplementedError

    def check_processed(self):
        if not os.path.exists(self._images_dir):
            logging.error(f'[{self._name}] {self._images_dir} NOT exists!')
            raise NotADirectoryError
        if len([i for i in os.listdir(self._images_dir) if os.path.splitext(i)[-1] in self._valid_image_suffix]) <= 0:
            logging.error(f'[{self._name}] {self._images_dir} contains NO images!')
            raise FileNotFoundError
        if not os.path.exists(self._images_bin_fn):
            logging.error(f'[{self._name}] {self._images_bin_fn} NOT exists!')
            raise FileNotFoundError
        if not os.path.exists(self._cameras_bin_fn):
            logging.error(f'[{self._name}] {self._cameras_bin_fn} NOT exists!')
            raise FileNotFoundError
    
class _3DScannerDatasetAdaptor(_AbstractDatasetAdaptor):

    def check_unprocessed(self):
        file_infos = {}
        obj_fn = os.path.join(self._search_dir, 'export.obj')
        for fn in os.listdir(self._search_dir):
            name, suffix = os.path.splitext(fn)
            if name in ['thumb_00000']:
                continue
            if suffix in self._valid_image_suffix:
                json_fn = os.path.join(self._search_dir, f'{name}.json')
                if os.path.exists(json_fn):
                    file_infos[name] = {
                        'json': json_fn,
                        'image': os.path.join(self._search_dir, fn)
                    }
                else:
                    logging.error(f'[{self._name}] {json_fn} NOT exists!')
                    raise FileNotFoundError
        if os.path.exists(obj_fn):
            self._verts = load_obj(obj_fn)
        else:
            raise FileNotFoundError
        self._file_infos = file_infos
        self._ply_path = os.path.join(self._sparse_save_dir, 'points3D.ply')
        if len(self._file_infos) <= 0:
            logging.error(f'[{self._name}] found NO valid data!')
            raise FileNotFoundError
        if not os.path.exists(self._ply_path):
            logging.error(f'[{self._name}] {self._ply_path} NOT exists!')
            raise FileNotFoundError

    def _internal_call(self):
        camera_type = 'PINHOLE'
        Images = {}
        Cameras = {}
        # sort name via frame idxes since may use temporal information in the future
        sorted_names = sorted(self._file_infos.keys(), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        suffix = None

        TransARKit2Colmap = np.array([
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        for idx, (name) in tqdm(enumerate(sorted_names)):
            camera_id = int(idx+1)
            info = self._file_infos[name]
            image_fn = info['image']
            if suffix is None:
                suffix = os.path.splitext(image_fn)[-1]
            # Processing cameras.bin
            image = load_image(image_fn)
            H, W, _ = image.shape
            meta = load_json(info['json'])
            Cameras[int(idx+1)] = Camera(
                id=camera_id, model=camera_type, width=W, height=H, 
                params=[meta['intrinsics'][0], meta['intrinsics'][4], meta['intrinsics'][2], meta['intrinsics'][5]]
            )
            
            # Processing images.bin, xys and point3D_ids is empty since useless
            transform = np.array(meta['cameraPoseARFrame']).reshape([4, 4])
            # transform = RotY(np.pi) @ transform

            qvec = rotmat2qvec(transform[:3, :3])
            tvec = transform[:3, -1]

            Images[int(idx+1)] = Image(
                    id=int(idx+1), qvec=qvec, tvec=tvec, camera_id=camera_id,
                    name=os.path.basename(image_fn),
                    xys=[], 
                    point3D_ids=[]
                )
            
        write_images_binary(Images, self._images_bin_fn)
        write_cameras_binary(Cameras, self._cameras_bin_fn)
        # Processing points3D.ply
        num_pts = len(self._verts)
        shs = np.random.random((num_pts, 3)) / 255.0
        # shs = np.ones((num_pts, 3))
        # pointcloud in 3d scanner is in right-hand
        V = self._verts
        # V = (RotY(180 / 180 * np.pi) @ RotX(180 / 180 * np.pi)[:3, :3] @ self._verts.T).T
        storePly(self._ply_path, V, SH2RGB(shs) * 255)

        # storePly(self._ply_path, self._verts, SH2RGB(shs) * 255)
        
        # Copying images
        os.system(f'cp {self._search_dir}/*{suffix} {self._images_dir}/')
        logging.info(f'[{self._name}] Completed!')


class _ColmapDatasetAdaptor(_AbstractDatasetAdaptor):

    def check_unprocessed(self):
        pass
    
    def _internal_call(self):
        os.system(f'bash bash/3dscanner_to_colmap.sh {self._search_dir} {self._save_dir}')
        self._ply_bin_fn = os.path.join(self._sparse_save_dir, "points3D.bin")
        self._ply_path = os.path.join(self._sparse_save_dir, 'points3D.ply')
        if not os.path.exists(self._ply_bin_fn):
            logging.error(f'{self._ply_bin_fn} NOT found!')
            raise FileNotFoundError
        xyz, rgb, _ = read_points3D_binary(self._ply_bin_fn)
        storePly(self._ply_path, xyz, rgb)

class _ARKitColmapDatasetAdaptor(_AbstractDatasetAdaptor):

    def check_unprocessed(self):
        pass

    def _internal_call(self):
        os.system(f'python3 utils/toolkit/convert_arkit_colmap.py --dataset_dir {self._search_dir} --save_dir {self._save_dir}')


