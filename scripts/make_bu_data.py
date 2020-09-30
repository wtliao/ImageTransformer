from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
import glob

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='/data/scene_understanding/bottom-up-feature/adaptive/test2014', help='downloaded feature directory')
parser.add_argument('--output_dir', default='/data/scene_understanding/bottom-up-feature/adaptive/test2014/cocobu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
# infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
#            'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',
#            'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
#            'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']

infiles = ['test2014_resnet101_faster_rcnn_genome.tsv.0',
           'test2014_resnet101_faster_rcnn_genome.tsv.1',
           'test2014_resnet101_faster_rcnn_genome.tsv.2']
corrupted_ims = [300104, 147295, 321486]
# infiles = ['test2015_resnet101_faster_rcnn_genome.tsv']

if not os.path.exists(args.output_dir + '_att'):
    os.makedirs(args.output_dir + '_att')
if not os.path.exists(args.output_dir + '_fc'):
    os.makedirs(args.output_dir + '_fc')
if not os.path.exists(args.output_dir + '_box'):
    os.makedirs(args.output_dir + '_box')

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+t") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            i +=1
            item['image_id'] = int(item['image_id'])
            if item['image_id'] in corrupted_ims:
                continue
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field].encode()),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            if not os.path.exists(os.path.join(args.output_dir + '_att', str(item['image_id']))):
                np.savez_compressed(os.path.join(args.output_dir + '_att', str(item['image_id'])), feat=item['features'])
            if not os.path.exists(os.path.join(args.output_dir + '_fc', str(item['image_id']))):
                np.save(os.path.join(args.output_dir + '_fc', str(item['image_id'])), item['features'].mean(0))
            if not os.path.exists(os.path.join(args.output_dir + '_box', str(item['image_id']))):
                np.save(os.path.join(args.output_dir + '_box', str(item['image_id'])), item['boxes'])
            if i%5000==0:
                print(i)
