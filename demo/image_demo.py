# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

import os
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import cv2
from scipy.optimize import leastsq

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette

def apply_color_map(mask):
    # 빈 RGB 이미지를 생성
    color_map = {
        0: [128, 0, 0],      # 빨간색 계열 Road
        1: [255, 255, 0],    # 노랑색 Sidewalk
        2: [128, 128, 128],  # 회색 Construction
        3: [139, 69, 19],    # 갈색 Fence
        4: [128, 0, 128],    # 보라색 계열 Pole
        5: [255, 105, 180],  # 분홍색 Traffic light
        6: [128, 64, 0],     # 주황색 계열 Traffic sign
        7: [0, 128, 0],      # 초록색 Nature
        8: [135, 206, 235],  # 하늘색 Sky
        9: [128, 128, 64],   # 카키 계열 Person
        10: [128, 64, 128],  # 자주 계열 Rider
        11: [64, 128, 128],  # 터콰즈 계열 Car
        12: [0, 0, 0]  # 검정색 계열 배경
    }
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 각 클래스에 대한 색상을 적용
    for key, color in color_map.items():
        colored_mask[mask == key] = color
        
    return colored_mask

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def calculate_residuals(c, x, y):
    xi = c[0]
    yi = c[1]
    ri = c[2]
    return ((x-xi)**2 + (y-yi)**2 - ri**2)


def fisheye_vignetting(image_path,pred_mask):
    x_coords = []
    y_coords = []

    non_vignetting_threshold = 80
    inner_circle_margin = 30

    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(960,540))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Scan each row of the image
    for i in range(img_gray.shape[0]):

        # Scan from the left
        for j in range(img_gray.shape[1]):
            if np.any(img_gray[i,j] > non_vignetting_threshold):
                x_coords.append(j)
                y_coords.append(i)
                break

        # Scan from the right
        for j in range(img_gray.shape[1]-1, -1, -1):
            if np.any(img_gray[i,j] > non_vignetting_threshold):
                x_coords.append(j)
                y_coords.append(i)
                break

    # Convert the lists to numpy arrays
    x = np.array(x_coords)
    y = np.array(y_coords)

    # Initial guess for circle parameters (center at middle of image, radius half the image width)
    c0 = [img_gray.shape[1]/2, img_gray.shape[0]/2, img_gray.shape[1]/4]

    # Perform least squares circle fit
    c, _ = leastsq(calculate_residuals, c0, args=(x, y))

    img_color = img.copy()
    # Draw the circle on the original image
    x = cv2.circle(img_color, (int(c[0]), int(c[1])), int(c[2])-10, (0, 255, 0), 2)

    # Fill in the inside of the circle
    mask_valid = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    y = cv2.circle(mask_valid, (int(c[0]), int(c[1])), int(c[2])-inner_circle_margin, 1, -1)

    t = (y == 0)
    # pred_mask[t] = 12

    return t

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('test_dirs', help='Test Image dirs')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)

    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    test_img_paths = sorted(glob(args.test_dirs + '/*.png'))

    results = []
    for idx, img in tqdm(enumerate(test_img_paths)):
        # test a single image
        result = inference_segmentor(model, img)

        pred = result[0].astype(np.uint8)
        pred = Image.fromarray(pred) # 이미지로 변환
        pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
        pred = np.array(pred)

        if idx == 0:
            t = fisheye_vignetting(img,pred)
            pred[t] = 12
        else: pred[t] = 12
        for class_id in range(12):
            class_mask = (pred == class_id).astype(np.uint8)
            if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                mask_rle = rle_encode(class_mask)
                results.append(mask_rle)
            else: # 마스크가 존재하지 않는 경우 -1
                results.append(-1)

        if idx <= 20: 
            colored_pred = apply_color_map(pred)
            img = Image.fromarray(colored_pred)
            img = img.resize((960, 540), Image.NEAREST)
            img.save(f'submission/mask_{idx}.png')

        # show the results
        # if idx <= 10:
        #     file, extension = os.path.splitext(img)
        #     pred_file = f'{file}_pred{extension}'
        #     assert pred_file != img
        #     model.show_result(
        #         img,
        #         result,
        #         palette=get_palette(args.palette),
        #         out_file=pred_file,
        #         show=False,
        #         opacity=args.opacity)

    submit = pd.read_csv('data/sample_submission.csv')
    submit['mask_rle'] = results
    submit.to_csv('submission/submit_0913_hrda.csv', index=False)


if __name__ == '__main__':
    main()
