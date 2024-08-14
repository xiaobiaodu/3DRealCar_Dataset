import sys
sys.path.append('.')
import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.io import load_image
from utils.logging import logging
from concurrent.futures import ThreadPoolExecutor
from utils.utility import masked_psnr, masked_ssim, create_window

def blur_entropy_metric(fn):
    try:
        name = os.path.splitext(os.path.basename(fn))[0]
        image = load_image(fn)
        mask = (image.sum(-1) != 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # BLUR
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mean, var = cv2.meanStdDev(laplacian[mask])
        # ENTROPY
        num_bins = 256
        hist = cv2.calcHist([gray[mask]], [0], None, [num_bins], [0, num_bins])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        max_entropy = np.log2(num_bins)
        normalized_entropy = entropy / max_entropy
        return name, var[0,0], normalized_entropy
    except Exception as e:
        logging.error(f'failed to calculate psnr and ssim for {name} since {e}!')
        return None

def multi_calculate_unpaired_metrices(search_dir, max_workers=64, mode='v2'):
    jobs = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    for root, _, fns in os.walk(search_dir):
        for fn in fns:
            if '.jpg' not in fn:
                continue
            jobs.append(executor.submit(blur_entropy_metric, os.path.join(root, fn)))
    names, blurs, entropies = zip(*[job.result() for job in tqdm(jobs) if job.result() is not None])
    output = {}
    for name, blur, entropy in zip(names, blurs, entropies):
        output[name] = {
            'blur': blur,
            'entropy': entropy
        }
    executor.shutdown()
    return output

def psnr_ssim_metric(real_fn, pred_fn, mask_fn, mode='v2'):
    try:
        name = os.path.splitext(os.path.basename(real_fn))[0]
        real_img = torch.Tensor(load_image(real_fn)).cuda() / 255
        pred_img = torch.Tensor(load_image(pred_fn)).cuda() / 255
        mask = torch.Tensor(load_image(mask_fn)).cuda()[:, :, 0, None] / 255
        real_img = real_img.permute([2, 0, 1])
        pred_img = pred_img.permute([2, 0, 1])
        if mode == 'v1':
            mask = (mask.permute([2, 0, 1]) > 0).float()
        elif mode == 'v2':
            mask = (mask.permute([2, 0, 1]) > 240 / 255).float()
        real_img = real_img * mask
        psnr, _ = masked_psnr(real_img, pred_img, mask=mask)
        ssim, _ = masked_ssim(real_img, pred_img, mask=mask)
        return name, psnr.detach().cpu().numpy(), ssim.detach().cpu().numpy()
    except Exception as e:
        logging.error(f'failed to calculate psnr and ssim for {name} since {e}!')
        return None

def multi_calculate_paired_metrices(search_dir, max_workers=64, mode='v2'):
    pred_dir = f'{search_dir}/pred'
    real_dir = f'{search_dir}/real'
    mask_dir = f'{search_dir}/mask'
    fns = os.listdir(pred_dir)
    fns = sorted(fns, key=lambda x:int(os.path.splitext(x)[0].split('_')[-1]))
    jobs = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    for fn in fns:
        if '.jpg' not in fn:
            continue
        real_fn = os.path.join(real_dir, fn)
        pred_fn = os.path.join(pred_dir, fn)
        mask_fn = os.path.join(mask_dir, fn)
        jobs.append(executor.submit(psnr_ssim_metric, real_fn, pred_fn, mask_fn, mode=mode))
    results = [job.result() for job in tqdm(jobs) if job.result() is not None]
    if len(results) == 0:
        return None
    names, psnrs, ssims = zip(*results)
    output = {}
    for name, psnr, ssim in zip(names, psnrs, ssims):
        output[name] = {
            'psnr': psnr,
            'ssim': ssim
        }
    executor.shutdown()
    return output

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_names_list', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--test_name', type=str)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--mode', type=str, choices=['v1', 'v2'], required=True, default='v1')
    parser.add_argument("--run_unpaired", action='store_true')
    parser.add_argument("--run_tests", action='store_true')
    parser.add_argument("--save_log", action='store_true')
    return parser.parse_args()

def main():
    logging.setLevel(logging.INFO)
    args = get_arguments()
    save_dir = args.save_dir
    exp_name = args.exp_name
    models_dir = args.models_dir
    mode = args.mode
    save_log = args.save_log
    max_workers=args.max_workers
    test_name = args.test_name
    run_tests = args.run_tests
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, os.path.dirname(args.exp_name)), exist_ok=True)
    logging.addFileHandler(os.path.join(save_dir, 'eval.log'))
    if args.dataset_name:
        dataset_names = [args.dataset_name]
    elif args.dataset_names_list:
        dataset_names = [i.strip() for i in open(args.dataset_names_list).readlines()]
    else:
        dataset_names = os.listdir(models_dir)
    paired_report = os.path.join(save_dir, f'{exp_name}.{mode}.paired')
    paired_report_detail = os.path.join(save_dir, f'{exp_name}.{mode}.paired_detail')
    unpaired_report = os.path.join(save_dir, f'{exp_name}.{mode}.unpaired')
    unpaired_report_detail = os.path.join(save_dir, f'{exp_name}.{mode}.unpaired_detail')
    total_psnrs = []
    total_ssims = []
    total_blurs = []
    total_entropies = []
    if save_log:
        f_paired = open(paired_report, 'w')
        f_paired_detail = open(paired_report_detail, 'w')
        f_unpaired = open(unpaired_report, 'w')
        f_unpaired_detail = open(unpaired_report_detail, 'w')
        f_paired.write(f'DatasetName\tPSNR(Mean)\tSSIM(Mean)\n')
        f_paired_detail.write(f'DatasetName\tImageName\tPSNR\tSSIM\n')
        f_unpaired.write(f'DatasetName\tBLUR(Mean)\tENTROPY(Mean)\n')
        f_unpaired_detail.write(f'DatasetName\tImageName\tBLUR\tENTROPY\n')
    valid_dataset_names = []
    for dataset_name in dataset_names:
        model_dir = f'{models_dir}/{dataset_name}/{exp_name}'
        if not os.path.exists(model_dir):
            logging.warn(f'Skipped {dataset_name} since UNTRAINED!')
            if save_log:
                f_paired.write('\n')
                f_unpaired.write('\n')
            continue
        if not os.path.exists(os.path.join(model_dir, 'test')):
            logging.warn(f'Skipped {dataset_name} since UNTESTED!')
            if save_log:
                f_paired.write('\n')
                f_unpaired.write('\n')
            continue
        if run_tests:
            base_dir = f'{model_dir}/tests/{test_name}'
        else:
            base_dir = f'{model_dir}/test'
        logging.info(f'start testing model under {base_dir}')
        psnr_ssim_results = multi_calculate_paired_metrices(
            search_dir=os.path.join(base_dir, 'paired_data/images'), 
            max_workers=max_workers, mode=mode)
        if psnr_ssim_results is None:
            if save_log:
                f_paired.write('\n')
                f_unpaired.write('\n')
            continue
        psnrs = []
        ssims = []
        names = []
        for k, v in psnr_ssim_results.items():
            psnr = v['psnr']
            ssim = v['ssim']
            psnrs.append(psnr)
            ssims.append(ssim)
            info = f'{dataset_name}\t{k}\t{psnr}\t{ssim}'
            if save_log:
                f_paired_detail.write(f'{info}\n')
                f_paired_detail.flush()
        info = f'{dataset_name}\t{np.mean(psnrs)}\t{np.mean(ssims)}'
        logging.info(f'PSNR&SSIM: {info}')
        if save_log:
            f_paired.write(f'{info}\n')
            f_paired.flush()
        total_psnrs.append(np.mean(psnrs))
        total_ssims.append(np.mean(ssims))
        valid_dataset_names.append(dataset_name)
        if args.run_unpaired:
            blur_entropy_results = multi_calculate_unpaired_metrices(
                search_dir=os.path.join(base_dir, 'unpaired_data/images'), 
                max_workers=max_workers, mode=mode)
            blurs = []
            entropies = []
            for k, v in blur_entropy_results.items():
                blur = v['blur']
                entropy = v['entropy']
                blurs.append(blur)
                entropies.append(entropy)
                info = f'{dataset_name}\t{k}\t{blur}\t{entropy}'
                if save_log:
                    f_unpaired_detail.write(f'{info}\n')
                    f_unpaired_detail.flush()
            info = f'{dataset_name}\t{np.mean(blurs)}\t{np.mean(entropies)}'
            logging.info(f'BLUR&ENTROPY: {info}')
            if save_log:
                f_unpaired.write(f'{info}\n')
                f_unpaired.flush()
            total_blurs.append(np.mean(blurs))
            total_entropies.append(np.mean(entropies))
    assert len(valid_dataset_names) == len(total_psnrs)
    if len(total_psnrs) > 0:
        if not args.run_unpaired:
            for dataset_name, psnr, ssim in zip(valid_dataset_names, total_psnrs, total_ssims):
                logging.info(f'{exp_name}\t{dataset_name}\t{psnr}\t{ssim}\n')
            logging.info(f'TOTAL_AVG\t{np.mean(total_psnrs)}\t{np.mean(total_ssims)}\n')
        else:
            for dataset_name, psnr, ssim, blur, entropy in zip(valid_dataset_names, total_psnrs, total_ssims, total_blurs, total_entropies):
                logging.info(f'{exp_name}\t{dataset_name}\t{psnr}\t{ssim}\t{blur}\t{entropy}\n')
            logging.info(f'TOTAL_AVG\t{np.mean(total_psnrs)}\t{np.mean(total_ssims)}\t{np.mean(total_blurs)}\t{np.mean(total_entropies)}\n')
    if save_log:
        f_paired.close()
        f_paired_detail.close()
        f_unpaired.close()
        f_unpaired_detail.close()

if __name__ == '__main__':
    main()






