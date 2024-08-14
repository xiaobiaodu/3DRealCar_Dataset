import sys
sys.path.append('submodules/gaussian_splatting')
import os
import cv2
import json
import yaml
import pickle
import numpy as np
import open3d as o3d
from PIL import Image
from easydict import EasyDict
from utils.logging import logging
from utils.util import fetchPly, read_extrinsics_binary, read_intrinsics_binary

def easydict2dict(easy_dict):
    if isinstance(easy_dict, EasyDict):
        return {k: easydict2dict(v) for k, v in easy_dict.items()}
    elif isinstance(easy_dict, dict):
        return {k: easydict2dict(v) for k, v in easy_dict.items()}
    elif isinstance(easy_dict, (list, tuple)):
        return [easydict2dict(item) for item in easy_dict]
    else:
        return easy_dict
    
# Opencv Image IO
def load_image(fn, mode='cv'):
    # Reading Images in RGB format
    assert mode in ['cv', 'pil']
    if mode == 'cv':
        image = np.array(cv2.imread(fn))
        if len(image.shape) == 3:
            image = image[:, :, ::-1].copy()
    else:
        image = Image.open(fn)
        exif = image._getexif()
        if exif:
            orientation = exif.get(0x0112)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
        image = np.array(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    return image

def save_image(fn, obj, mode='cv'):
    assert mode in ['cv']
    if len(obj.shape) == 2:
        obj = obj[:, :, None]
    return cv2.imwrite(fn, obj[:, :, ::-1])

# Json IO
def load_json(fn):
    return json.load(open(fn))

def save_json(fn, obj):
    json.dump(obj, open(fn, 'w'))

def load_numpy(fn):
    return np.load(fn)

def save_numpy(fn, obj):
    np.save(fn, obj)

# Object IO
def load_obj(fn):
    # TODO: use pytorch3d in the future
    verts = []
    for line in open(fn).readlines():
        line = line.strip()
        if line[0] != '#':
            splited_line = line.split()
            if splited_line[0] == 'v':
                verts.append([float(i) for i in splited_line[1:]])
    verts = np.array(verts)
    return verts

# Pickle IO
def load_pickle(fn):
    return pickle.load(open(fn, 'rb'))

def save_pickle(fn, obj):
    pickle.dump(obj, open(fn, 'wb'))

# Yaml IO
def load_yaml(fn):
    return EasyDict(yaml.safe_load(open(fn).read()))

def save_yaml(fn, obj):
    with open(fn, 'w') as f:
        yaml.dump(easydict2dict(obj), f, default_flow_style=False)

# Camera Trace IO, work with LiNRE
def load_camera_trace(fn):
    title = 'ImageName\tCameraModelName\tQuaternion\tTranslation\tFocalLengthX\tFocalLengthY\tImageHeight\tImageWidth'
    camera_trace_infos = []
    for lidx, line in enumerate(open(fn).readlines()):
        if lidx == 0:
            if line.strip() != title:
                logging.error(f'Invalid title "{line.strip()}", expected "{title}"!')
                raise ValueError
        else:
            line = line.strip()
            if len(line) > 0:
                image_name, camera_model, Q, T, FocalX, FocalY, Height, Width = line.strip().split('\t')
                Q = np.array([float(i) for i in Q.split()])
                T = np.array([float(i) for i in T.split()])
                if len(Q) != 4:
                    logging.error(f'Invalid quaternion: {Q}!')
                    raise ValueError
                if len(T) != 3:
                    logging.error(f'Invalid translation: {T}!')
                    raise ValueError
                FocalX = float(FocalX)
                FocalY = float(FocalY)
                Height = float(Height)
                Width = float(Width)
                camera_trace_infos.append([
                    image_name, camera_model, Q, T, FocalX, FocalY, Height, Width
                ])
            else:
                logging.warn(f'[line {lidx}] contains NOTHING, skipped!')
    return camera_trace_infos

def save_camera_trace(fn, cam_infos_sorted, cam_intrinsics, valid_names=None):
    title = 'ImageName\tCameraModelName\tQuaternion\tTranslation\tFocalLengthX\tFocalLengthY\tImageHeight\tImageWidth'
    with open(fn, 'w') as f:
        f.write(title + '\n')
        for _, cam_info in cam_infos_sorted:
            name = os.path.splitext(cam_info.name)[0]
            if valid_names is not None and name not in valid_names:
                continue
            cam = cam_intrinsics[cam_info.camera_id]
            qvec = ' '.join([str(i) for i in cam_info.qvec])
            tvec = ' '.join([str(i) for i in cam_info.tvec])
            f.write(f'{name}\t{cam.model}\t{qvec}\t{tvec}\t{cam.params[0]}\t{cam.params[1]}\t{cam.height}\t{cam.width}\n')
        logging.info(f'Write camera info into "{fn}"')

def load_colmap(dataset_dir):
    ply_file = os.path.join(dataset_dir, "sparse/0", "points3D.ply")
    ply_data = fetchPly(ply_file)
    init_pcd = ply_data.points
    init_color = ply_data.colors
    init_normal = ply_data.normals
    cameras_extrinsic_file = os.path.join(dataset_dir, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(dataset_dir, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_infos_sorted = sorted(cam_extrinsics.items(), key=lambda x: x[1].name)
    return init_pcd, init_color, init_normal, cam_intrinsics, cam_infos_sorted

# Open3D IO
def load_mesh(fn):
    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(fn))
    return mesh_o3d
    # return o3d.t.geometry.PointCloud.from_legacy(o3d.io.read_point_cloud(fn))

def save_mesh(fn, mesh):
    o3d.io.write_triangle_mesh(fn, mesh)
