import json
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

coco = json.load(open('cocotalk.json'))
im_info = coco['images']
exa_ind = np.random.randint(120000,size=(100))
print(len(im_info))

for i in range(100):
    temp = im_info[exa_ind[i]]
    bbid = str(temp['id'])
    impath = temp['file_path']
    im = imread('/media/Data/Sen/coco_cap/'+impath)
    bb = np.load('./new_cocobu_box/'+bbid+'.npy')
    #print(bb)
    print(bb.shape)
    print(im.shape)
    plt.imshow(im)
    for j in range(bb.shape[0]):
        plt.gca().add_patch(plt.Rectangle((bb[j,0],bb[j,1]),bb[j,2]-bb[j,0],bb[j,3]-bb[j,1],fill=False,edgecolor='blue',linewidth=2,alpha=0.5))
    plt.show()
