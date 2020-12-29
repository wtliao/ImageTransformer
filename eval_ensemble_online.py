from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils_h as eval_utils
from eval_online import eval_online
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--ids', nargs='+', required=False, help='id of the models to ensemble')
parser.add_argument('--weights', nargs='+', required=False, default=None, help='id of the models to ensemble')
# parser.add_argument('--models', nargs='+', required=True
#                 help='path to model to evaluate')
# parser.add_argument('--infos_paths', nargs='+', required=True, help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()

model_infos = []
model_paths = []


opt.split = 'online_test'  # online_test or val
opt.test_online = 1
save_results = 1
opt.language_eval = 0
opt.beam_size = 3
opt.batch_size = 100

if opt.split == 'val':
    assert not opt.test_online, "for val opt.test_online must be 1"
if opt.test_online:
    opt.input_fc_dir = 'data/adaptive/test2014/cocobu_fc'
    opt.input_att_dir = 'data/adaptive/test2014/cocobu_att'
    opt.input_box_dir = 'data/adaptive/test2014/cocobu_box'
    opt.input_flag_dir = 'data/tmp/cocobu_flag_h_v1_challenging14'
    opt.input_label_h5 = None
    assert opt.split == 'online_test', "assert wrong split"
else:
    opt.input_flag_dir = 'data/tmp/cocobu_flag_h_v1'


# ensemble should be the same model structure but different training processes only
aoa_id = '3d1'
opt.caption_model = 'aoa' + aoa_id

aoa_num = 3

opt.id = 'h_v' + aoa_id
# model_ids = ['57', 'best', 'best', 62, 63, '61', 65, 65]  # list(range(51, 55)) + ['best']
# append_info = ['_rl', '_new2_rl_40', '_new3_rl', '_refine_val_rl','_refine_test_rl','_new4_rl', '_new5_all_rl', '_new6_all_rl']
# model_ids = [57, 61, 41, 75, 61, 69, 69, 65, 65]  # list(range(51, 55)) + ['best']
# append_info = ['_rl', '_new2_rl_40', '_new2_rl', '_new3_rl', '_new4_rl', '_refine_val_rl', '_refine_test_rl','_new6_all_rl', '_new5_all_rl']
# model_ids = [59,        'best',        59,           'best',      55,             59,                    67,                  67,             75,         66,                 66]  # list(range(51, 55)) + ['best']
# append_info = ['_rl', '_new2_rl_40','_new2_25_rl','_new3_25_rl','_new4_25_rl', '_new9_new1_25_rl', '_new10_new1_37_rl', '_new11_new1_40_rl', '_refine_val_rl','_new5_all_rl', '_new6_all_rl']
model_ids = [59,                    59,                 'best',         55,         67,             66] 
append_info = ['_new9_new1_25_rl', '_new2_25_rl', '_new3_25_rl', '_new4_25_rl','_new10_new1_37_rl', '_new6_all_rl']
print("============================================")
print("=========beam search size:{}=================".format(opt.beam_size))

opt.ids = []
for model_id, app in zip(model_ids, append_info):
    model_infos.append(utils.pickle_load(open('log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/infos_{}.pkl'.format(opt.id, aoa_num, app, model_id), 'rb')))
    model_paths.append('log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/model_{}.pth'.format(opt.id, aoa_num, app, model_id))
    opt.ids.append(opt.id + app + str(model_id))

# Load one infos
infos = model_infos[0]

# override and collect parameters
replace = ['input_json', 'batch_size', 'id']
for k in replace:
    setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))

vars(opt).update({k: vars(infos['opt'])[k] for k in vars(infos['opt']).keys() if k not in vars(opt)})  # copy over options from model


opt.use_box = max([getattr(infos['opt'], 'use_box', 0) for infos in model_infos])
assert max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]), 'Not support different norm_att_feat'
assert max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]), 'Not support different norm_box_feat'

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
from models.AttEnsemble2 import AttEnsemble

_models = []
for i in range(len(model_infos)):
    model_infos[i]['opt'].start_from = None
    model_infos[i]['opt'].vocab = vocab
    tmp = models.setup(model_infos[i]['opt'])
    tmp.load_state_dict(torch.load(model_paths[i]))
    _models.append(tmp)

if opt.weights is not None:
    opt.weights = [float(_) for _ in opt.weights]
model = AttEnsemble(_models, weights=opt.weights)
model.seq_length = opt.max_length
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

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

if opt.split == 'val':
    _info = json.load(open(opt.input_json))
    split_ix = {'train': [], 'val': [], 'test': []}
    for ix in range(len(_info['images'])):
        img = _info['images'][ix]
        if 'split' not in img:
            split_ix['val'].append(ix)
        elif not img['split'] == 'train':
            split_ix['val'].append(ix)
    loader.split_ix = split_ix


opt.id = '+'.join([_ + 'w' + str(__) for _, __ in zip(opt.ids, model.weights.cpu().data.numpy())])
print("==================={}===================".format(opt.id))
# Set sample options
if opt.split == 'online_test':
    predictions = eval_online(model, crit, loader, vars(opt))
else:
    loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

#import ipdb; ipdb.set_trace()
if opt.split == 'online_test':
    cache_path1 = os.path.join('online_eval_results', opt.id + '_' + 'bs' + str(opt.beam_size) + '_test.json')
    cache_path2 = os.path.join('results', 'captions_test2014_ToT_results.json')
    corrupted_ims = [300104, 147295, 321486]
    for c_img in corrupted_ims:
        predictions.append({'image_id':c_img, 'caption':''})
else:
    cache_path1 = os.path.join('online_eval_results', opt.id + '_' + 'bs' + str(opt.beam_size) + '_val.json')
    cache_path2 = os.path.join('results', 'captions_val2014_ToT_results.json')


with open(cache_path1, 'w') as f:
    json.dump(predictions, f)

with open(cache_path2, 'w') as f:
    json.dump(predictions, f)
print("Write prediction results to {}".format(cache_path1))

print("=================================================================")
print("===================Ensemble Evalluation {} DONE!======================".format(opt.ids))
print("=================================================================")
