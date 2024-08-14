import os
import tempfile
from utils.io import save_image
from utils.logging import logging

class ImageCollector:

    def __init__(self, name, save_dir, keep_tmp=False, fps=25):
        self._name = name
        self._save_dir = save_dir
        self._keep_tmp = keep_tmp
        self._fps = fps
        self._image_dir = tempfile.TemporaryDirectory().name
        self._fidx = 0
        os.makedirs(self._image_dir)
        if self._keep_tmp:
            logging.info(f'images will be saved in {self._image_dir}')
    
    def collect(self, image, finalize=False):
        save_image(f'{self._image_dir}/{self._fidx}.jpg', image)
        self._fidx += 1
        if finalize:
            self.finalize()
    
    def finalize(self):
        save_name = f'{self._save_dir}/{self._name}.mp4'
        cmd = f'bash bash/image_to_video.sh {self._image_dir} {save_name} {self._fps}'
        os.system(cmd)
        if not self._keep_tmp:
            cmd = f'rm -rf {self._image_dir}/*.jpg'
            os.system(cmd)
        else:
            logging.info(f'Temporal images are saved in {self._image_dir}!')
        logging.info(f'Video saved in {os.path.abspath(save_name)}!')