
import sys
import torch
import random
import kornia
import open3d
import numpy as np
from utils.io import *
from matplotlib import cm
from utils.util import focal2fov, getWorld2View2

def set_random_seed(seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    
def RotX(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
def RotY(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def RotZ(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def HomoRotX(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    
def HomoRotY(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def HomoRotZ(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def slerp(q1, q2, t):
    cos_half_theta = q1 @ q2
    if cos_half_theta < 0.0:
        cos_half_theta = -cos_half_theta
        q2 = -q2
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1 - cos_half_theta ** 2)
    if np.abs(sin_half_theta) > 0.001:
        one_over_sin_half_theta = 1.0 / sin_half_theta
        a = np.sin(1.0 - t) * half_theta * one_over_sin_half_theta
        b = np.sin(t) * half_theta * one_over_sin_half_theta
    else:
        a = 1.0 - t
        b = t
    qo = a * q1 + b * q2
    return qo / np.linalg.norm(qo)

def get_projection_matrix_numpy(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def get_full_projection_matrix_numpy(R, T, FoVx, FoVy, 
                                     translate=np.array([0., 0., 0.]), znear=0.01, zfar=100.0):
    world_view_transform = getWorld2View2(R, T, translate=translate).T
    projection_matrix = get_projection_matrix_numpy(
        znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).T
    full_proj_transform = world_view_transform @ projection_matrix
    return full_proj_transform

def get_FovXY(camera_model, fx, fy, W, H):
    if camera_model=="SIMPLE_PINHOLE":
        FovY = focal2fov(fx, H)
        FovX = focal2fov(fx, W)
    elif camera_model=="PINHOLE":
        FovY = focal2fov(fy, H)
        FovX = focal2fov(fx, W)
    else:
        logging.error(f'Unsupport camera model "{camera_model}"!')
        raise NotImplementedError
    return FovX, FovY

def points_to_image_space(points, camera_model, R, T, fx, fy, W, H, translate=np.array([0., 0., 0.])):
    FovX, FovY = get_FovXY(
        camera_model=camera_model, 
        fx=fx, fy=fy, 
        W=W, H=H)
    proj = get_full_projection_matrix_numpy(
        R=R, T=T, translate=translate, FoVx=FovX, FoVy=FovY)
    points = np.pad(points.copy(), [[0, 0], [0, 1]], constant_values=1.0) @ proj
    points = points[:, :3] / points[:, -1, None]
    points[:, :3] = points[:, :3] / points[:, -1, None]
    points[:, 0] = ((points[:, 0] + 1.0) * W - 1.0) / 2
    points[:, 1] = ((points[:, 1] + 1.0) * H - 1.0) / 2
    return points

def check_inside_triangle(p, v1, v2, v3):
    AB = v2 - v1
    BC = v3 - v2
    CA = v1 - v3
    AP = p - v1
    BP = p - v2
    CP = p - v3
    cross_product_AB_AP = np.cross(AB, AP)
    cross_product_BC_BP = np.cross(BC, BP)
    cross_product_CA_CP = np.cross(CA, CP)
    if (cross_product_AB_AP >= 0 and cross_product_BC_BP >= 0 and cross_product_CA_CP >= 0) or \
    (cross_product_AB_AP <= 0 and cross_product_BC_BP <= 0 and cross_product_CA_CP <= 0):
        return True
    else:
        return False

def find_outlier(points, n_neighbours=20, std_ratio=2.0):
    # points: N x 3
    dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    sorted_ind = np.argsort(dist, axis=1)
    sorted_dist = np.take_along_axis(dist, indices=sorted_ind, axis=1)
    flag = np.std(sorted_dist[:, :n_neighbours], -1) > std_ratio
    return flag

def check_points_in_image(point, W, H, border=0):
    x, y = point
    x = int(np.round(x))
    y = int(np.round(y))
    if x >= border and x < W - border and \
        y >= border and y < H - border:
        return True
    return False

def visualize_2d_points(image_points, W, H, flags=None, alpha=None, point_width=5):
    image = np.zeros([H, W, 3])
    if alpha is not None:
        if alpha.shape[0] != H or alpha.shape[1] != W:
            logging.error(f'Mismatched shape between image and alpha, got [{H}, {W}, 3] and {alpha.shape} expectively!')
            raise RuntimeError
        image[:, :, 0] = alpha[:, :, 0].copy()
    if flags is not None:
        if len(flags) != len(image_points):
            logging.error(f'Mismatched size between image points and flags, got {len(image_points)} and {len(flags)} respectively!')
    for idx, (x, y) in enumerate(image_points[:, :2]):
        if check_points_in_image([x, y], W, H):
            x = int(np.round(x))
            y = int(np.round(y))
            image[y-point_width:y+point_width, x-point_width:x+point_width, 0] = 0
            image[y-point_width:y+point_width, x-point_width:x+point_width, 1] = 255
            image[y-point_width:y+point_width, x-point_width:x+point_width, 2] = 0
            if flags is not None:
                if flags[idx]:
                    image[y-point_width:y+point_width, x-point_width:x+point_width, 0] = 0
                    image[y-point_width:y+point_width, x-point_width:x+point_width, 1] = 0
                    image[y-point_width:y+point_width, x-point_width:x+point_width, 2] = 255
    return image

def normalize_tensor(tensor):
    if tensor is None:
        return tensor
    max_val = tensor.max()
    min_val = tensor.min()
    return (tensor - min_val) / torch.clamp(max_val - min_val, 1e-8)

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = np.array(colormap.colors)
    image_long = np.array(image * 255, dtype=np.int32)
    image_long_min = np.min(image_long)
    image_long_max = np.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(np.min(depth)) if min is None else min
    far_plane = float(np.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = np.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image

# Mesh-Depth Rasterization
def mesh_to_depth(mesh_o3d, intr, extr, H, W, dep_max=300.0):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)
    rays_mesh = scene.create_rays_pinhole(
        intrinsic_matrix=intr, 
        extrinsic_matrix=np.linalg.inv(extr), 
        width_px=W, height_px=H)
    rays_rast = scene.cast_rays(rays_mesh)
    depth_rast = rays_rast['t_hit'].numpy()
    alpha = (depth_rast==np.inf)
    depth_rast[alpha] = 0.0
    norm_rast = rays_rast['primitive_normals'].numpy()
    return depth_rast[None], np.transpose(norm_rast, [2, 0, 1]), alpha[None] * 1.0

def clean_mesh(mesh_o3d, bbox):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    points = mesh_o3d.vertex.positions.numpy()
    colors = mesh_o3d.vertex.colors.numpy()
    normals = mesh_o3d.vertex.normals.numpy()
    triangles = mesh_o3d.triangle.indices.numpy()
    mask = (points[:, 2] >= zmin)
    triangles = mesh_o3d.triangle.indices.numpy()
    N = triangles.shape
    triangles_mask = mask[triangles]
    triangles_mask = (triangles_mask.sum(-1) > 0)
    return mesh_o3d.select_faces_by_mask(triangles_mask)

# Sampling Ellipse Camera Positions
def ellipse(N, a, b):
    points = []
    for theta in np.linspace(-np.pi, np.pi, N):
        points.append([a*np.cos(theta), b*np.sin(theta)])
    return np.array(points)

def generate_ellipse_positions(X=5, Y=3, Zspan=[-0.5, 3], Nxy=50, Nz=20):
    total_points = []
    Z = np.max(np.abs(Zspan))
    zs = np.linspace(Zspan[0], Zspan[1], Nz)
    if len(zs) > 1:
        zs = zs[:-1]
    for z in zs:
        y = Y * np.cos(np.arcsin(z / Z))
        x = y / Y * X
        xy_pts = ellipse(Nxy, x, y)
        xy_pts = np.pad(xy_pts, [[0, 0], [0, 1]])
        xy_pts[:, -1] = z    
        total_points.append(xy_pts)
    total_points = np.concatenate(total_points, axis=0)
    return total_points

def lookat_transform(CamPos, Point):
    v3 = np.array(Point - CamPos)
    v3 = v3 / np.linalg.norm(v3)
    v = np.array([0, 0, -1])
    v1 = np.cross(v, v3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(v3, v1)
    v2 = v2 / np.linalg.norm(v2)
    R = np.eye(4)
    R[:3, :3] = np.array([v1, v2, v3]).T
    R[:3, -1] = CamPos
    R = np.linalg.inv(R)
    return R

# Remove points given LiGSCamera
def find_outside_alpha_v1(camera, points):
    proj = camera.full_proj_transform.cuda()
    inv_alpha = (camera.original_alpha[0].cuda() < 0.5).bool()
    W = camera.image_width
    H = camera.image_height
    N = points.shape[0]
    device = points.device
    ones = torch.ones((N, 1)).to(device)
    points = torch.cat([points, ones], dim=-1)
    image_points = torch.mm(points, proj)
    image_points = image_points[:, :3] / image_points[:, -1, None]
    image_points[:, 0] = ((image_points[:, 0] + 1.0) * W - 1.0) / 2
    image_points[:, 1] = ((image_points[:, 1] + 1.0) * H - 1.0) / 2
    image_points = torch.round(image_points).long()
    in_image_flag = torch.logical_and(image_points[:, 0] >= 0, \
                torch.logical_and(image_points[:, 0] < W, \
                torch.logical_and(image_points[:, 1] >= 0, image_points[:, 1] < H)))
    remap_indices = torch.arange(0, N).to(device).long()[in_image_flag]
    valid_image_points = image_points[in_image_flag]
    # if camera.crop_bbox is not None:
    #     valid_image_points[:, 0] -= camera.crop_bbox[0]
    #     valid_image_points[:, 1] -= camera.crop_bbox[1]
    flatten_indices = valid_image_points[:, 0] + valid_image_points[:, 1] * inv_alpha.shape[1]
    inv_alpha = inv_alpha.reshape([-1])[flatten_indices]
    out_alpha_flag = torch.tensor([False] * N, dtype=torch.bool).to(device)
    out_alpha_flag[remap_indices] = inv_alpha
    prune_mask = torch.logical_and(out_alpha_flag, in_image_flag)
    return prune_mask

def find_outside_alpha_v2(camera, points, border=1):
    "V2: support border for alpha pruning to overcome boundary ambiguity"
    N_repeat = border * 2 + 1
    proj = camera.full_proj_transform.cuda()
    out_alpha = (camera.original_alpha[0].cuda() < 0.5).bool()
    W = camera.image_width
    H = camera.image_height
    N = points.shape[0]
    device = points.device
    ones = torch.ones((N, 1)).to(device)
    points = torch.cat([points, ones], dim=-1)
    image_points = torch.mm(points, proj)
    image_points = image_points[:, :3] / image_points[:, -1, None]
    image_points[:, 0] = ((image_points[:, 0] + 1.0) * W - 1.0) / 2
    image_points[:, 1] = ((image_points[:, 1] + 1.0) * H - 1.0) / 2
    image_points = torch.round(image_points).long()
    out_alpha_flag = torch.tensor([True] * N, dtype=torch.bool).to(device)
    for i, ni in enumerate(range(N_repeat)):
        for j, nj in enumerate(range(N_repeat)):
            image_points_tmp = image_points.clone()
            image_points_tmp[:, 0] += ni
            image_points_tmp[:, 1] += nj
            in_image_flag = torch.logical_and(image_points_tmp[:, 0] >= 0, \
                        torch.logical_and(image_points_tmp[:, 0] < W, \
                        torch.logical_and(image_points_tmp[:, 1] >= 0, image_points_tmp[:, 1] < H)))
            remap_indices = torch.arange(0, N).to(device).long()[in_image_flag]
            valid_image_points = image_points_tmp[in_image_flag]
            # if camera.crop_bbox is not None:
            #     valid_image_points[:, 0] -= camera.crop_bbox[0]
            #     valid_image_points[:, 1] -= camera.crop_bbox[1]
            flatten_indices = valid_image_points[:, 0] + valid_image_points[:, 1] * out_alpha.shape[1]
            out_alpha_tmp = out_alpha.reshape([-1])[flatten_indices]
            alpha_flag = torch.tensor([False] * N, dtype=torch.bool).to(device)
            alpha_flag[remap_indices] = out_alpha_tmp
            out_alpha_flag = torch.logical_and(out_alpha_flag, alpha_flag)
    prune_mask = torch.logical_and(out_alpha_flag, in_image_flag)
    return prune_mask

# Color Space Conversion
def RGB2LAB(rgb):
    assert len(rgb.shape) == 3
    # Assume batch equals to 1
    rgb = rgb[None]
    assert rgb.shape[1] == 3
    if rgb.max() > 1:
        logging.error(f'RGB2LAB assume input image be in [0, 1]')
        raise ValueError
    if type(rgb) not in [torch.Tensor, torch.tensor]:
        rgb = torch.Tensor(rgb)
    lab = kornia.color.rgb_to_lab(rgb)[0]
    l = lab[0] / 100
    a = lab[1]
    b = lab[2]
    return l, a, b

def get_image_stats(image, alpha=None):
    if alpha is not None:
        assert image.shape == alpha.shape
        pixels = image.reshape(-1)[alpha.reshape(-1)]
    else:
        pixels = image.reshape(-1)
    if type(pixels) in [torch.Tensor, torch.tensor]:
        mean = torch.mean(pixels)
        median = torch.median(pixels)
        std = torch.std(pixels)
        mean = mean.cpu().numpy()
        median = median.cpu().numpy()
        std = std.cpu().numpy()
    else:
        mean = np.mean(pixels)
        median = np.median(pixels)
        std = np.std(pixels)
    return {
        'mean': float(mean),
        'median': float(median),
        'std': float(std)
    }

