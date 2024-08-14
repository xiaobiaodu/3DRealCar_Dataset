import os, sys
sys.path.append('.')
sys.path.append(os.getcwd())
import cv2
import copy
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import supervision as sv
import matplotlib.pyplot as plt
from utils.logging import logging
from torchvision.ops import box_convert
from groundingdino.util import box_ops
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from segment_anything import build_sam, SamPredictor 
from groundingdino.util.inference import annotate, predict
from utils.io import load_yaml, save_pickle, load_image, save_image, save_numpy
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def load_image_transformed(image_path):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = load_image(image_path, mode='pil')
    image_transformed, _ = transform(Image.fromarray(image), None)
    return image, image_transformed

def setup():
    # ======================== Load Grounding DINO model ========================
    logging.info(f'[ SAM ] Load Grounding DINO model')
    def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        logging.info(f"[ SAM ] Model loaded from {cache_file} => {log}")
        _ = model.eval()
        return model   
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    global groundingdino_model 
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    # ======================== Load Segment Anything model ========================
    logging.info(f'[ SAM ] Load SAM model')
    sam_checkpoint = 'resources/models/sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    global sam_predictor
    sam_predictor = SamPredictor(sam)

def find_max_box(boxes):
    max_size = -1
    max_idx = 0
    debug = []
    for idx, box in enumerate(boxes):
        size = box[2] * box[3]
        debug.append(size)
        if size > max_size:
            max_idx = idx
            max_size = size
    return max_idx

def segment_with_text_prompt(images_lists, text_prompt, output_dir, skip_list, TEXT_PROMPT, BOX_TRESHOLD=0.3, TEXT_TRESHOLD=0.25):
    save_dir = output_dir
    skips = []
    if skip_list not in ['', None]:
        if os.path.exists(skip_list):
            skips = [i.strip() for i in open(skip_list).readlines()]
    for image_path in tqdm(images_lists):
        image_source, image = load_image_transformed(image_path)
        name = os.path.basename(image_path)
        if name in skips:
            logging.info(f'[ SAM ] Skipped {image_path} since found in skip list')
            save_image(os.path.join(save_dir, f'{name}'), np.zeros(image_source.shape))
            continue
        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )
        logging.info(f'[ SAM ] Detecting {boxes.shape[0]} boxed of {text_prompt} in {image_path}')
        if boxes.shape[0] == 0:
            continue
        idx = find_max_box(boxes)          
        boxes = boxes[idx,None]
        logits = logits[idx,None]
        phrases = [phrases[idx]]
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        sam_predictor.set_image(image_source)
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
        masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        torch.cuda.empty_cache()
        mask_final = torch.zeros_like(masks[0, 0]).bool()
        for mask in masks[:, 0]:
            mask_final = mask_final | mask.bool()
        # __import__('ipdb').set_trace()
        def show_mask(mask, image, random_color=True):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            annotated_frame_pil = Image.fromarray(image).convert("RGBA")
            mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
            return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
        annotated_frame_with_mask = show_mask(mask_final.cpu().numpy(), annotated_frame)
        mask = mask_final.cpu().numpy()
        os.makedirs(save_dir, exist_ok=True)
        image_base_path = os.path.splitext(os.path.basename(image_path))[0]
        # output_annotated_frame_with_mask = os.path.join(save_dir, f'{image_base_path}_vis.jpg')
        # save_image(output_annotated_frame_with_mask, annotated_frame_with_mask)
        # output_mask = os.path.join(save_dir, f'{image_base_path}.jpg')
        # save_image(output_mask, mask * 255)
        save_numpy(os.path.join(save_dir, f'{image_base_path}.npy'), mask)

        meta = {
            'image_shape': mask.shape[:2],
            'boxes_xyxy': boxes_xyxy,
            'boxes': boxes,
            'logits': logits,
            'phrases': phrases
        }
        save_pickle(os.path.join(save_dir, f'{image_base_path}.pkl'), meta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', required=True, type=str)
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--skip_list', type=str)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()
    os.makedirs(args.dataset_dir, exist_ok=True)
    logging.addFileHandler(os.path.join(args.dataset_dir, 'segmentation.log'))
    hparams = load_yaml(args.yaml)
    setup()
    image_files = glob(args.dataset_dir + f"/{hparams.TrainDatasetSetting.feature_settings.image.dir}/*.jpg") 
    image_files += glob(args.dataset_dir + f"/{hparams.TrainDatasetSetting.feature_settings.image.dir}/*.png")  
    segment_with_text_prompt(
        images_lists=image_files, 
        text_prompt=hparams.TrainDatasetSetting.segmentation_prompt, 
        output_dir=os.path.join(args.dataset_dir, hparams.TrainDatasetSetting.feature_settings.alpha.dir),
        skip_list=args.skip_list,
        TEXT_PROMPT=hparams.TrainDatasetSetting.segmentation_prompt,
        BOX_TRESHOLD=args.box_threshold,
        TEXT_TRESHOLD=args.text_threshold
    )
    os.system(f'touch {args.dataset_dir}/.segmentation')

