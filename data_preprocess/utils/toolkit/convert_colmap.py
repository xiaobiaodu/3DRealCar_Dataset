import sys
sys.path.append('.')
import os
import shutil
import argparse
from tqdm import tqdm
from utils.logging import logging
from utils.io import load_image, save_image
from utils.util import storePly, read_points3D_binary

def get_arguments():
    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = argparse.ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--mask_path", "-m", type=str, default='')
    parser.add_argument("--mask_skip_list", type=str, default='')
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--camera", default="PINHOLE", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    return  parser.parse_args()

def colmap_process(colmap_command, source_path, mask_path, camera, use_gpu):

    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    logging.info('[ COLMAP ] Start colmap Feature extraction ...')
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    if mask_path not in ['', None]:
        feat_extracton_cmd += f" --ImageReader.mask_path {mask_path}"
        logging.info(f'[ COLMAP ] Using mask when extracting features')
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"[ COLMAP ] Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    logging.info('[ COLMAP ] Start colmap Feature matching ...')
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"[ COLMAP ] Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    logging.info('[ COLMAP ] Start colmap Bundle adjustment ...')
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path "  + source_path + "/input \
        --output_path "  + source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"[ COLMAP ] Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    if len(os.listdir(source_path + "/distorted/sparse")) != 1:
        logging.error('[ COLMAP ] Failed')
        return False
    
    logging.info('[ COLMAP ] Start copy files ...')
    files = os.listdir(source_path + "/distorted/sparse/0")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    for file in files:
        source_file = os.path.join(source_path, "distorted/sparse/0", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    ply_bin_fn = f'{source_path}/sparse/0/points3D.bin'
    ply_fn = f'{source_path}/sparse/0/points3D.ply'
    xyz, rgb, _ = read_points3D_binary(ply_bin_fn)
    storePly(ply_fn, xyz, rgb)
    return True

def main():
    args = get_arguments()
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0
    retried_times = 0
    logging.info(f'[ COLMAP ] Start processing images for colmap, remove exif info')
    for fn in tqdm(os.listdir(os.path.join(args.source_path, 'input'))):
        if os.path.splitext(fn)[-1] in ['.jpg', '.png']:
            img = load_image(os.path.join(args.source_path, 'input', fn), mode='pil')
            save_image(os.path.join(args.source_path, 'input', fn), img)
    mask_save_path = None
    if args.mask_path not in ['', None]:
        skips = []
        if args.mask_skip_list not in ['', None]:
            skips = [i.strip() for i in open(args.mask_skip_list).readlines()]
        mask_save_path = args.mask_path + '_colmap'
        os.makedirs(mask_save_path, exist_ok=True)
        logging.info(f'[ COLMAP ] Start processing masks for colmap')
        for root, _, fns in os.walk(args.mask_path):
            for fn in fns:
                if '.jpg' in fn and 'vis' not in fn:
                    mask = load_image(os.path.join(root, fn))
                    if fn in skips:
                        mask = mask * 0 + 1
                    if args.invert_mask:
                        mask = 255 - mask
                    save_image(os.path.join(mask_save_path, fn+'.png'), mask)
        logging.info(f'[ COLMAP ] Saved colmap mask into {mask_save_path}')
        logging.info(f'[ COLMAP ] Done')
    while True:
        succ = colmap_process(
            colmap_command=colmap_command, source_path=args.source_path, 
            mask_path=mask_save_path, camera=args.camera, use_gpu=use_gpu)
        if succ:
            os.makedirs(f'{args.source_path}/images', exist_ok=True)
            os.system(f'ln -s {os.path.abspath(args.source_path)}/input/* {args.source_path}/images')
            logging.info('[ COLMAP ] Done')
            break
        retried_times += 1
        os.system(f'rm -rf {args.source_path}/distorted')
        logging.error(f'[ COLMAP ] Retried {retried_times} times, Colmap failed, start to rebuild ...')

if __name__ == '__main__':
    main()

