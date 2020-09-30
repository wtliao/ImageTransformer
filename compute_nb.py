import json
import numpy as np


def compute_iou(bbox):
    N = bbox.shape[0]
    flag = np.zeros((N, N))
    # np.fill_diagonal(iou,0)
    width = bbox[:, 2] - bbox[:, 0]
    ht = bbox[:, 3] - bbox[:, 1]
    area = width * ht
    for i in range(N):
        temp = bbox[i, :]
        t_left, t_right, t_top, t_bot = temp[0], temp[2], temp[1], temp[3]
        # x_left = np.zeros(N)
        # x_right = np.zeros(N)
        # x_top = np.zeros(N)
        # x_bot = np.zeros(N)
        for j in range(N):

            x_left = max(t_left, bbox[j, 0])
            x_top = max(t_top, bbox[j, 1])
            x_right = min(t_right, bbox[j, 2])
            x_bot = min(t_bot, bbox[j, 3])

            intersection_area = (x_right - x_left) * (x_bot - x_top)
            self_area = (t_right - t_left) * (t_bot - t_top)
            iou = intersection_area / self_area
            iou1 = intersection_area / area[j]
            if x_right < x_left or x_bot < x_top:
                continue
            if iou >= 0.1 or iou1 >= 0.9:
                flag[i, j] = 1
    np.fill_diagonal(flag, 1)
    return flag


coco = json.load(open('/home/liao/work_code/AoANet/data/tmp/4/cocotalk.json'))
print(coco.keys())
im_info = coco['images']

for i in range(len(im_info)):
    print(i)

    temp = im_info[i]
    bbid = str(temp['id'])
    bb = np.load('/home/liao/work_code/AoANet/data/adaptive/cocobu_box/' + bbid + '.npy')
    flag = compute_iou(bb)
    np.save('/home/liao/work_code/AoANet/data/tmp/cocobu_flag/' + bbid + '.npy', flag)
