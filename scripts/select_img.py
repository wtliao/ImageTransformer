# show image, gt caption and generated caption together for comparison

import os
import json
import shutil 
import glob
# import misc.utils as utils

# img_list = [219928]
# coco_path = '/home/liao/work_code/AoANet/data/coco2014'
coco_path = '/home/liao/work_code/AoANet/eval_results/qualitative/tmp'
# image_name = json.load(open(coco_path + '/annotations/image_name_val2014.json'))

save_to = '/home/liao/work_code/AoANet/eval_results/qualitative/tmp/attention_qualitative/'


box_path = '/home/liao/work_code/AoANet/data/adaptive/cocobu_box'
img_list = []
files = glob.glob('/home/liao/work_code/AoANet/eval_results/qualitative/tmp/attention/*.jpg')
for f in files:
    img_name = os.path.basename(f)
    source_path = os.path.join(coco_path, 'test', img_name)
    if not os.path.exists(source_path):
        source_path = os.path.join(coco_path, 'val', img_name)
    to_path = os.path.join(save_to, img_name)
    shutil.copyfile(source_path, to_path)

# for f in files:
#     img_id = os.path.basename(f).split('.')[0]
#     img_list.append(img_id)

# for i in img_list:
#     img_name = image_name[str(i)]
#     img_path = os.path.join(coco_path, 'val', image_name[str(i)])
#     if not os.path.exists(img_path):
#         img_path = os.path.join(coco_path, 'test', image_name[str(i)])
    
#     shutil.copyfile(img_path, save_to + img_name)
