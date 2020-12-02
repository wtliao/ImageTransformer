import json
import numpy as np
import os
import glob


def compute_iou(bbox):
    N = bbox.shape[0]
    flag = np.zeros((N, N)) + 2
    width = bbox[:, 2] - bbox[:, 0]
    ht = bbox[:, 3] - bbox[:, 1]
    area = width * ht
    for i in range(N):
        temp = bbox[i, :]
        t_left, t_right, t_top, t_bot = temp[0], temp[2], temp[1], temp[3]
        for j in range(N):

            x_left = max(t_left, bbox[j, 0])
            x_top = max(t_top, bbox[j, 1])
            x_right = min(t_right, bbox[j, 2])
            x_bot = min(t_bot, bbox[j, 3])

            intersection_area = (x_right - x_left) * (x_bot - x_top)
            self_area = (t_right - t_left) * (t_bot - t_top)
            iou = intersection_area / self_area
            iou1 = intersection_area / area[j]
            # if x_right < x_left or x_bot < x_top:
            #     continue
            # if iou >= 0.05:
            flag[i, j] = 2
            # if iou >= 0.9 :
            #     flag[i, j] = 1
            # if iou1 >= 0.9:
            #     flag[i, j] = 3
            # if iou >= 0.9 and iou1 >= 0.9:
            #     flag[i, j] = 2
            if iou >= 0.9 or iou1 >= 0.9:
                if iou > iou1:
                    flag[i, j] = 3
                elif iou < iou1:
                    flag[i, j] = 1
                else:
                    continue

    np.fill_diagonal(flag, 2)
    return flag

# def check_ij_ji(flag):
#     N, _ = flag.shape()
#     for i in range(N):
#         for j in range(N):
#             if flag[i,j] ==1:
#                 assert flag[j,i]==3, "if A is P of B, B muss be C of A"

# get adjacent matrix of challening data
# files = glob.glob('./data/adaptive/test2014/cocobu_box/*.npy')
# files = glob.glob('./data/adaptive/test2015/cocobu_box/*.npy')
files = glob.glob('./data/adaptive/cocobu_box/*.npy')
for i, f in enumerate(files):
    bb = np.load(f)
    flag = compute_iou(bb)
    # change the path where you would like to save the flag information
    np.save('/home/liao/work_code/AoANet/data/tmp/cocobu_flag/' + os.path.basename(f), flag) 
    if i % 5000 == 0:
        print("{}/{}".format(i, len(files)))