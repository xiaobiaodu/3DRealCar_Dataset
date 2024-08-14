import os
import sys
import shutil
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    return parser.parse_args()

def main():
    args = get_arguments()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    cameras = [i for i in os.listdir(dataset_dir) if 'Camera_' in i]
    filenames = None
    static_name = '000000.jpg'
    collected = [f'{camera}/{static_name}' for camera in cameras]
    for cidx, camera in enumerate(cameras):
        if filenames is None:
            filenames = [i for i in os.listdir(os.path.join(dataset_dir, camera)) if os.path.splitext(i)[-1] == '.jpg' and i != static_name]
            filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
        collected += [f'{camera}/{fn}' for i, fn in enumerate(filenames) if i % len(cameras) == cidx]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'static.txt'), 'w') as f:    
        for fidx, fn in enumerate(collected):
            new_name = f'frame_{fidx:06d}.jpg'
            if static_name in fn:
                f.write(f'{new_name}\n')
            shutil.copy(os.path.join(dataset_dir, fn), os.path.join(save_dir, new_name))

if __name__ == '__main__':
    main()


