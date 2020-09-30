from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import numpy as np

import time
from six.moves import cPickle
import ipdb

import models
import opts1 as opts
# from dataloader_relative import *
from dataloader1 import *
import misc.utils2 as utils
# import misc.utils as utils
from eval_online import eval_online
from dataloaderraw import *
import argparse
import torch


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
# parser.add_argument('--model', type=str, default='log/tmp/train_ours/log_refine_aoa_{}_aoa3_new/model_22.pth',
#                     help='path to model to evaluate')
# parser.add_argument('--cnn_model', type=str, default='resnet101',
#                     help='resnet101, resnet152')
# parser.add_argument('--infos_path', type=str, default='log/tmp/train_ours/log_refine_aoa_{}_aoa3_new/infos_22.pkl',
#                     help='path to infos to evaluate')

parser.add_argument('--model', type=str, default='log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/model_{}.pth',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/infos_{}.pkl',
                    help='path to infos to evaluate')
# parser.add_argument('--model_id', type=str, default=None, help='which specifi chech point to load')
# parser.add_argument('--append_info', type=str, default=None, help="such as old/new")

opts.add_eval_options(parser)

opt = parser.parse_args()
# opt.dump_images = 1
# opt.dump_json = 1
# opt.num_images = 1
opt.language_eval = 1
opt.beam_size = 3
opt.batch_size = 100
opt.split = 'online_test'  # online_test
opt.test_online = 1
# opt.use_val = 1
# opt.use_test = 1


aoa_id = '3d1'
aoa_num = 3
append_info = '_new6_all_rl'
opt.caption_model = 'aoa' + aoa_id
opt.id = 'h_v' + aoa_id
opt.input_flag_dir = 'data/tmp/cocobu_flag_h_v1'
model_ids = [61]  # list(range(21, 31))
best_cider = -1
best_epoch = -1

if opt.test_online:
    opt.input_fc_dir = 'data/adaptive/test2014/cocobu_fc'
    opt.input_att_dir = 'data/adaptive/test2014/cocobu_att'
    opt.input_box_dir = 'data/adaptive/test2014/cocobu_box'
    opt.input_flag_dir = 'data/tmp/cocobu_flag_h_v1_challenging14'
    opt.input_label_h5 = None
    assert opt.split == 'online_test', "assert wrong split"

print("============================================")
print("=========beam search size:{}=================".format(opt.beam_size))
print("============================================")

for model_id in model_ids:
    opt.model = 'log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/model_{}.pth'.format(opt.id, aoa_num, append_info, model_id)
    opt.infos_path = 'log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/infos_{}.pkl'.format(opt.id, aoa_num, append_info, model_id)
    test_result = {}
    # opt.model = 'log/tmp/train_ours/log_refine_aoa_h_v3_old_aoa3/model_best.pth'
    # opt.infos_path = 'log/tmp/train_ours/log_refine_aoa_h_v3_old_aoa3/infos_best.pkl'
    # Load infos
    print("Evluation using infor_path:{}".format(opt.infos_path))
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
        iteration = infos.get('iter', 0)
        epoch = infos.get('epoch', 0)
        print("=====start from {} epoch-- {} iterations=============".format(epoch, iteration))
        print("=====refine aoa: {}   ==========".format(infos['opt'].refine_aoa))
        print("=====learning rate decay every: {}   ==========".format(infos['opt'].learning_rate_decay_every))
        # caption_model = getattr(infos['opt'], 'caption_model','')
        # print("caption model: {}".format(caption_model))

    # override and collect parameters
    # replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir',
    #            'input_label_h5', 'input_json', 'batch_size', 'id']
   
    ignore = ['start_from']

    replace = ['input_json', 'batch_size', 'id']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if k not in vars(opt):
                # copy over options from model
                vars(opt).update({k: vars(infos['opt'])[k]})

    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    print("Evluation using model:{}".format(opt.model))
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    model.eval()
    crit = utils.LanguageModelCriterion()
    # ipdb.set_trace()
    # Create the Data Loader instance
    if len(opt.image_folder) == 0:
        loader = DataLoader(opt)
    else:
        loader = DataLoaderRaw({'folder_path': opt.image_folder,
                                'coco_json': opt.coco_json,
                                'batch_size': opt.batch_size,
                                'cnn_model': opt.cnn_model})
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']

    # ipdb.set_trace()
    # Set sample options
    opt.datset = opt.input_json
    predictions = eval_online(model, crit, loader, vars(opt))
    
    corrupted_ims = [300104, 147295, 321486]
    # caption_corrupted_ims = ['a plane flies in the blue sky',
    #                          'a man sits before a church',
    #                          'a green lemon on the pizza']

    for c_img in corrupted_ims:
        predictions.append({'image_id':c_img, 'caption':''})
    
    cache_path = os.path.join('results', 'captions_test2014_ToT_results.json')
    with open(cache_path, 'w') as f:
        json.dump(predictions, f)
    print("Write the standard submit results file to {}".format(cache_path))
    print("=====================test online Done!=============")

