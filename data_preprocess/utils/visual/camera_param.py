import cv2
import numpy as np
from utils.visual.video import ImageCollector
from utils.utility import points_to_image_space
from utils.util import qvec2rotmat


def visualize_cam_params(points, cam_intrinsics, cam_infos_sorted, 
                         save_dir, name, keep_tmp=False, point_width=5):
    image_collector = ImageCollector(name=name, save_dir=save_dir, keep_tmp=keep_tmp)
    for sidx, (_, cam_info) in enumerate(cam_infos_sorted):
        cam = cam_intrinsics[cam_info.camera_id]
        image = np.zeros((cam.height, cam.width, 3))
        image_points = points_to_image_space(
            points=points, camera_model=cam.model, 
            R=qvec2rotmat(cam_info.qvec).T, T=cam_info.tvec,
            fx=cam.params[0], fy=cam.params[1], 
            W=cam.width, H=cam.height)
        for x, y in image_points[:, :2]:
            x = int(x)
            y = int(y)
            if x >= 0 and x < cam.width and y > 0 and y < cam.height:
                image[y-point_width:y+point_width, x-point_width:x+point_width] = 255
        cv2.putText(image, cam_info.name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 8)
        image_collector.collect(image=image, finalize=(sidx == len(cam_infos_sorted) - 1))




