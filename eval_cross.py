from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
import ipdb
import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import glob
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


write_summary = True


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


# def add_summary_values(writer, key, val_dict, iteration):
#     if writer:
#         writer.add_scalars(key, val_dict, iteration)


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='log/tmp/log_aoanet/model.pth',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='log/tmp/log_aoanet/infos_aoanet.pkl',
                    help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()


data_list = ['','_sct','sal3_02','sal3','_tr_','_sml4_1','sml4','sml61','sml6']
not_list = ['sal2','sal1','sal0','_sml_']

for boxset_index in data_list:
      
    if '_sct' in boxset_index:
        boxset_index = '_sct'
    elif 'sal3_02' in boxset_index:
        boxset_index = '3_02'
    elif 'sal3' in boxset_index:
        boxset_index = '3'
    elif 'sal2' in boxset_index:
        boxset_index = '2'
    elif 'sal1' in boxset_index:
        boxset_index = '1'
    elif 'sal0' in boxset_index:
        boxset_index = '0'
    elif '_tr_' in boxset_index:
        boxset_index = '_tr'
    elif '_sml_' in boxset_index:
        boxset_index = '_sml'
    elif '_sml4_1' in boxset_index:
        boxset_index = '_sml4_1'
    elif 'sml4' in boxset_index:
        boxset_index = '_sml4'
    elif 'sml61' in boxset_index:
        boxset_index = '_sml61'
    elif 'sml6' in boxset_index:
        boxset_index = '_sml6'

    print("box set: {}".format(boxset_index))
    if len(boxset_index) > 0:
        opt.input_fc_dir = 'data/tmp/new{}_cocobu_fc'.format(boxset_index)
        opt.input_att_dir = 'data/tmp/new{}_cocobu_att'.format(boxset_index)
        opt.input_box_dir = 'data/tmp/new{}_cocobu_box'.format(boxset_index)

    # Load infos
    print("==================================================")
    print("Evluation using infor_path:{}".format(opt.infos_path))
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

        iteration = infos.get('iter', 0)
        epoch = infos.get('epoch', 0)
        print("Model trained on {} epoch -- {} iterations".format(epoch, iteration))

    # override and collect parameters
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir',
            'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

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
    print("==================================================")
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']

    # ipdb.set_trace()
    # Set sample options
    opt.datset = opt.input_json
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

    print('loss: ', loss)
    # if lang_stats:
    #     print(lang_stats)
    #     model_name = os.path.basename(opt.model).split('.')[0]
    #     result_file = '_'.join(opt.model.split("/")[-3:-1] + [str(opt.beam_size), model_name, '.txt'])
    #     with open("eval_results/{}".format(result_file), 'w') as f:
    #         f.writelines(lang_stats)
    #     print("Write evalation results to {}".format("eval_results/{}".format(result_file)))

    # if opt.dump_json == 1:
    #     # dump the json
    #     model_name = os.path.basename(opt.model).split('.')[0]
    #     result_file = '_'.join(opt.model.split("/")[-3:-1] + [str(opt.beam_size), model_name])
    #     json.dump(split_predictions, open('vis/tmp/vis-{}.json'.format(result_file), 'w'))
    #     print("Dump the json to {}".format('vis/tmp/vis-{}.json'.format(result_file)))
