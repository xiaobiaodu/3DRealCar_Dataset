import sys
sys.path.append('.')
import os
import json
import argparse
import numpy as np
from utils.io import load_obj
from utils.utility import RotX, RotY, RotZ
from utils.visual.camera import CameraPoseVisualizer

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_dir', type=str, required=True)
    parser.add_argument('--save_fn', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 5])
    search_dir = args.search_dir
    Trans = np.array([
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    for fn in os.listdir(search_dir):
        if '.json' in fn:
            if fn in ['annotations.json', 'info.json']:
                continue
            json_fn = os.path.join(search_dir, fn)
            data = json.load(open(json_fn))
            cameraPoseARFrame = np.array(data['cameraPoseARFrame']).reshape([4, 4])
            cameraPoseARFrame[:3, :3] = cameraPoseARFrame[:3, :3] @ RotX(np.pi)
            cameraPoseARFrame = Trans @ cameraPoseARFrame
            visualizer.extrinsic2pyramid(cameraPoseARFrame, 'r', 0.1, 1)
        elif fn == 'export.obj':
            verts = load_obj(os.path.join(search_dir, fn))
            visualizer.add_pointcloud(verts=verts[::100] @ Trans[:3, :3])
    visualizer.save(args.save_fn)

if __name__ == '__main__':
    main()


