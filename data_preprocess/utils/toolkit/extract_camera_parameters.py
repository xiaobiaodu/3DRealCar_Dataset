import sys
sys.path.append('.')
import os
import argparse
from utils.logging import logging
from utils.toolkit.core import LiGSToolkit
from utils.io import save_camera_trace, load_json

class CameraParameterExtractor(LiGSToolkit):

    def __init__(self, dataset_dir, save_dir, camera_json_fn=None, save_name='camera_track.txt'):
        super().__init__(dataset_dir=dataset_dir, save_dir=save_dir)
        self._save_name = save_name
        self._camera_json_fn = camera_json_fn
        self._valid_names = None
        if self._camera_json_fn is not None and self._camera_json_fn != '':
            self._valid_names = [param['img_name'] for param in load_json(self._camera_json_fn)]

    def run(self):
        os.makedirs(self._save_dir, exist_ok=True)
        save_camera_trace(
            os.path.join(self._save_dir, self._save_name), 
            cam_infos_sorted=self._cam_infos_sorted, 
            cam_intrinsics=self._cam_intrinsics,
            valid_names=self._valid_names)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--camera_json_fn', type=str)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_arguments()
    extractor = CameraParameterExtractor(
        args.dataset_dir, args.save_dir, 
        camera_json_fn=args.camera_json_fn,
        save_name=args.save_name)
    extractor.run()

if __name__ == '__main__':
    main()
