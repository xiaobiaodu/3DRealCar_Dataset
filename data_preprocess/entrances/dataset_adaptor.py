import os
import sys
sys.path.append('.')
import argparse
from utils.logging import logging
from utils.adaptor.dataset import _3DScannerDatasetAdaptor, \
    _ColmapDatasetAdaptor, _ARKitColmapDatasetAdaptor

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['3dscanner', 'colmap', 'arkit_colmap'], help="indicate running mode, should be in [3dscanner, colmap]")
    parser.add_argument('--search_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--suffix', type=str, nargs="+", default=['.jpg', '.png'])
    return parser.parse_args()

def main():
    args = get_arguments()
    mode = args.mode
    adaptor = None
    os.makedirs(args.save_dir, exist_ok=True)
    logging.addFileHandler(os.path.join(args.save_dir, 'dataset.log'))
    if mode == '3dscanner':
        adaptor = _3DScannerDatasetAdaptor(
        search_dir=args.search_dir, save_dir=args.save_dir, valid_image_suffix=args.suffix)
    elif mode == 'colmap':
        adaptor = _ColmapDatasetAdaptor(
        search_dir=args.search_dir, save_dir=args.save_dir, valid_image_suffix=args.suffix)
    elif mode == 'arkit_colmap':
        adaptor = _ARKitColmapDatasetAdaptor(
        search_dir=args.search_dir, save_dir=args.save_dir, valid_image_suffix=args.suffix)
    else:
        logging.error(f'Invalid mode "{mode}"')
        raise KeyError
    if adaptor is not None:
        adaptor()

if __name__ == '__main__':
    main()


