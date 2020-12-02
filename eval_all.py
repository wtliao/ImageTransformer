from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
import ipdb
import models
# import opts
import opts1
# from dataloader import *
from dataloader import *
# import misc.utils as utils
import misc.utils2 as utils
from dataloaderraw import *
# import eval_utils
import eval_utils_h as eval_utils
import argparse
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
parser.add_argument('--model', type=str, default='log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/model_{}.pth',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/infos_{}.pkl',
                    help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()

# specify parameter
opt.language_eval = 1
opt.batch_size = 100
opt.split = 'test'

aoa_id = 'b'
aoa_num = 3
append_info = ""
model_id = ""
opt.caption_model = 'aoa' + aoa_id
opt.id = 'h_v' + aoa_id
opt.input_flag_dir = 'data/tmp/cocobu_flag_{}'.format(opt.id)

opt.model = opt.model.format(opt.id, aoa_num, append_info)
opt.infos_path = opt.infos_path.format(opt.id, aoa_num, append_info)


if not os.path.isfile(opt.model) and os.path.exists(opt.model):
    model_list = glob.glob(opt.model + 'model*.pth')
    info_list = glob.glob(opt.model + 'infos_aoanet*.pkl')
else:
    model_list = [opt.model]
    info_list = [opt.infos_path]

model_root = os.path.dirname(model_list[0])
skip_epochs = ['best', 'model'] #+ [str(_) for _ in range(44)]

if write_summary:
    print("write summary to {}".format(model_root))
    tb_summary_writer = tb and tb.SummaryWriter(model_root, comment='test_model')

for model_path in model_list:
    if '_interrupt' in model_path:
        continue

    # fetch corresponding infos.pkl
    model_index = os.path.basename(model_path).split('.')[0].split('-')[-1]
    if model_index in skip_epochs:
        print("skpit {}".format(model_index))
        continue
    else:
        model_index = '-' + model_index

    infos_path = os.path.join(model_root, 'infos_aoanet' + model_index + '.pkl')
    assert infos_path in info_list, "infos_path does NOT exist!"
    # opt.dump_images = 1
    # opt.dump_json = 1
    # opt.num_images = 1
    # opt.language_eval = 1
    # opt.beam_size = 2
    # opt.batch_size = 100
    # opt.split = 'test'
    # opt.image_root = 'data/coco2014/'
    # # opt.image_folder = 'data/coco2014'
    boxset_index = ''
    
    if '_sct' in model_root:
        boxset_index = '_sct'
    elif 'sal3_02' in model_root:
        boxset_index = '3_02'
    elif 'sal3' in model_root:
        boxset_index = '3'
    elif 'sal2' in model_root:
        boxset_index = '2'
    elif 'sal1' in model_root:
        boxset_index = '1'
    elif 'sal0' in model_root:
        boxset_index = '0'
    elif '_tr_' in model_root:
        boxset_index = '_tr'
    elif '_sml_' in model_root:
        boxset_index = '_sml'
    elif '_sml4_1' in model_root:
        boxset_index = '_sml4_1'
    elif 'sml4' in model_root:
        boxset_index = '_sml4'
    elif 'sml4' in model_root:
        boxset_index = '_sml4'
    elif 'sml61' in model_root:
        boxset_index = '_sml61'
    elif 'sml6' in model_root:
        boxset_index = '_sml6'

    print("box set: {}".format(boxset_index))
    if len(boxset_index) > 0:
        opt.input_fc_dir = 'data/tmp/new{}_cocobu_fc'.format(boxset_index)
        opt.input_att_dir = 'data/tmp/new{}_cocobu_att'.format(boxset_index)
        opt.input_box_dir = 'data/tmp/new{}_cocobu_box'.format(boxset_index)

    # Load infos
    print("==================================================")
    print("Evluation using infor_path:{}".format(infos_path))
    with open(infos_path, 'rb') as f:
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
    print("Evluation using model:{}".format(model_path))
    model.load_state_dict(torch.load(model_path))
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
    opt.model = model_path
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))
    add_summary_value(tb_summary_writer, opt.split + '/test_loss', loss, epoch)

    bleu_dict = {}
    for k, v in lang_stats.items():
        if 'Bleu' in k:
            bleu_dict[k] = v
    if len(bleu_dict) > 0:
        tb_summary_writer.add_scalars(opt.split + '/Bleu', bleu_dict, epoch)

    for k, v in lang_stats.items():
        if 'Bleu' not in k:
            add_summary_value(tb_summary_writer, opt.split + '/' + k, v, epoch)

    print('loss: ', loss)
    if lang_stats:
        print(lang_stats)
        model_name = os.path.basename(opt.model).split('.')[0]
        result_file = '_'.join(opt.model.split("/")[-3:-1] + [str(opt.beam_size), model_name, '.txt'])
        with open("eval_results/{}".format(result_file), 'w') as f:
            f.writelines(lang_stats)
        print("Write evalation results to {}".format("eval_results/{}".format(result_file)))

    if opt.dump_json == 1:
        # dump the json
        model_name = os.path.basename(opt.model).split('.')[0]
        result_file = '_'.join(opt.model.split("/")[-3:-1] + [str(opt.beam_size), model_name])
        json.dump(split_predictions, open('vis/tmp/vis-{}.json'.format(result_file), 'w'))
        print("Dump the json to {}".format('vis/tmp/vis-{}.json'.format(result_file)))

tb_summary_writer.close()