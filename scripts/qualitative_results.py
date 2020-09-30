# show image, gt caption and generated caption together for comparison

import os
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

split = 'val'

# with open('./data/coco2014/annotations/captions_train2014.json', 'r') as f:
#     train = json.load(f)

info = json.load(open('./data/tmp/4/cocotalk.json'))

test_set = []
val_set = []
for i in info['images']:
    if i['split'] == 'test':
        test_set.append(i['id'])
    elif i['split'] == 'val':
        val_set.append(i['id'])
        
# h5_labels = h5py.File('~/work_code/AoANet/data/tmp/4/cocotalk_label.h5', 'r', driver='core')
# split_ix = {'val': [], 'test': []}
# pred = json.load(open('~/work_code/AoANet/eval_results/test/{}'.format(split)))
pred_results = json.load(open('/home/liao/work_code/AoANet/results/captions_val2014_ToT_results.json'))
pred_results_base = json.load(open('/home/liao/work_code/AoANet/online_eval_results/vb_aoa6_new_rl_bs3_40_val.json'))
coco_path = '/home/liao/work_code/AoANet/data/coco2014'

img_info = json.load(open(os.path.join(coco_path, 'annotations', 'captions_{}2014.json'.format(split))))
captions_gt = json.load(open(os.path.join(coco_path, 'annotations', 'captions_image_{}2014.json'.format(split))))
image_name = json.load(open(coco_path + '/annotations/image_name_val2014.json'))

save_to = '/home/liao/work_code/AoANet/eval_results/qualitative/val/'
# Write some Text

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.7
fontColor = (0, 0, 0)
lineType = 2
thickness = 1

with tqdm(total=len(pred_results)) as pbar:
    for k, p in enumerate(pred_results):
        img_id = p['image_id']
        if img_id in val_set:
            result = p['caption']
            result_b = pred_results_base[k]['caption']
            gt = captions_gt[str(img_id)]
            img_path = os.path.join(coco_path, 'val2014', image_name[str(img_id)])

            img = cv2.imread(img_path)
            h, w, _ = np.shape(img)
            img_plate = np.zeros((h, w + 400, 3), dtype=np.uint8) + 255
            img_plate[0:h, 0:w, :] = img

            img_plate = cv2.putText(img_plate, 'GT:', (w, 30), font,
                                    0.7, fontColor, thickness, cv2.LINE_AA)
            for i, g in enumerate(gt):
                img_plate = cv2.putText(img_plate, g['caption'], (w + 10, 50 + i * 20), font,
                                        0.4, fontColor, thickness, cv2.LINE_AA)
            
            img_plate = cv2.putText(img_plate, 'Baseline:', (w + 10, 100 + i * 20), font,
                                    0.7, (1, 190, 200), thickness, cv2.LINE_AA)
            img_plate = cv2.putText(img_plate, result_b, (w + 10, 120 + i * 20), font,
                                    0.4,(1, 190, 200), thickness, cv2.LINE_AA)

            img_plate = cv2.putText(img_plate, 'Ours:', (w + 10, 170 + i * 20), font,
                                    0.7, (255, 0, 0), thickness, cv2.LINE_AA)
            img_plate = cv2.putText(img_plate, result, (w + 10, 190 + i * 20), font,
                                    0.4, (255, 0, 0), thickness, cv2.LINE_AA)

            cv2.imwrite(save_to + image_name[str(img_id)], img_plate)
        pbar.update(1)
