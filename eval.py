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
from dataloader import *
import misc.utils2 as utils
# import misc.utils as utils
# import eval_utils_h as eval_utils
import eval_utils
from eval_online import eval_online
from dataloaderraw import *
import argparse
import torch
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

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
opt.split = 'test'
opt.test_online = 0
opt.use_val = getattr(opt, 'use_val', 0)
opt.use_test = getattr(opt, 'use_test', 0)
save_results = 0
val_and_test = 0

# if opt.use_test:
#     opt.split = 'val'

aoa_id = '3d1'
aoa_num = 3
append_info = '_best38_bs96_gpu8_rl'
opt.caption_model = 'aoa' + aoa_id
opt.id = 'h_v' + aoa_id
opt.caption_model = 'transformer'
opt.id = 'transformer'
opt.input_flag_dir = 'data/tmp/cocobu_flag_h_v1'
model_ids = ['best']#+list(range(70, 81))
best_cider = -1
best_epoch = -1
write_summary = False

print("============================================")
print("=========beam search size:{}=================".format(opt.beam_size))
print("=========test on {} set================".format(opt.split))
if write_summary:
    # summary_path = 'log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}'.format(opt.id, aoa_num, append_info)
    summary_path = 'log/tmp/train_ours/log_refine_{}'.format(opt.id)
    print("write test summary to {}".format(summary_path))
    tb_summary_writer = tb and tb.SummaryWriter(summary_path)
else:
    tb_summary_writer = None

for model_id in model_ids:
    # opt.model = 'log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/model_{}.pth'.format(opt.id, aoa_num, append_info, model_id)
    # opt.infos_path = 'log/tmp/train_ours/log_refine_aoa_{}_aoa{}{}/infos_{}.pkl'.format(opt.id, aoa_num, append_info, model_id)
    opt.model = 'log/tmp/train_ours/log_refine_{}/model_{}.pth'.format(opt.id, model_id)
    opt.infos_path = 'log/tmp/train_ours/log_refine_{}/infos_{}.pkl'.format(opt.id, model_id)
    test_result = {}

    # Load infos
    print("Evluation using infor_path:{}".format(opt.infos_path))
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
        iteration = infos.get('iter', 0)
        epoch = infos.get('epoch', 0)
        print("=====start from {} epoch-- {} iterations=============".format(epoch, iteration))
        print("=====refine aoa: {}   ==========".format(infos['opt'].refine_aoa))
        # print("=====aoa num: {}   ==========".format(infos['opt'].aoa_num))
        print("=====learning rate decay every: {}   ==========".format(infos['opt'].learning_rate_decay_every))
        # caption_model = getattr(infos['opt'], 'caption_model','')
        # print("caption model: {}".format(caption_model))

    # override and collect parameters
    # replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir',
    #            'input_label_h5', 'input_json', 'batch_size', 'id']

    ignore = ['start_from']

    if not opt.test_online:
        replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_flag_dir',
                    'input_label_h5', 'input_json', 'batch_size', 'id']
    else:
        replace = ['input_json', 'batch_size', 'id']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if k not in vars(opt):
                # copy over options from model
                vars(opt).update({k: vars(infos['opt'])[k]})


    # infos['opt'].use_val=0
    # infos['opt'].use_test=0

    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    opt.vocab = vocab
    # opt.caption_model = 'transformer'
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

    if val_and_test:
        loader.split_ix['val'] = loader.split_ix['test'] + loader.split_ix['val']
        loader.split_ix['test'] = loader.split_ix['val']
    # loader.split_ix['val'] = loader.split_ix['val'][:1000]
    # ipdb.set_trace()
    # Set sample options
    opt.datset = opt.input_json
    print(opt.language_eval)
    try:
        loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))
    except OSError:
        loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

    current_score = lang_stats['CIDEr']
    if best_cider < current_score:
        best_cider = current_score
        best_epoch = model_id

    if tb_summary_writer is not None:
        add_summary_value(tb_summary_writer, 'loss/test loss', loss, iteration)
        if lang_stats is not None:
            bleu_dict = {}
            for k, v in lang_stats.items():
                if 'Bleu' in k:
                    bleu_dict[k] = v
            if len(bleu_dict) > 0:
                tb_summary_writer.add_scalars('test/Bleu', bleu_dict, epoch)

            for k, v in lang_stats.items():
                if 'Bleu' not in k:
                    add_summary_value(tb_summary_writer, 'test/' + k, v, iteration)

    if save_results:
        model_name = opt.model.split('/')[-2].split('h_')[-1] + '_' + 'bs{}_{}'.format(opt.beam_size, model_id)
        cache_path = os.path.join('eval_results', opt.split, model_name + '.json')
        with open(cache_path, 'w') as f:
            json.dump(split_predictions, f)
        print("Write the results file to {}".format(cache_path))
    # if opt.dump_json == 1:
    #     # dump the json
    #     model_name = os.path.basename(opt.model).split('.')[0]
    #     result_file = '_'.join(['bs'+str(opt.beam_size), model_name])
    #     json.dump(split_predictions, open('vis/h/vis-{}.json'.format(result_file), 'w'))
    #     print("Dump the json to {}".format('vis/h/vis-{}.json'.format(result_file)))

    print("=================================================================")
    print("===================Evalluation {} DONE!======================".format(opt.model))
    print("===================Best {} Cider = {:.5f} in epoch {}!======================".format(opt.split, best_cider, best_epoch))
    print("======Best Val Cider = {:.4f} in epoch {}: iter {}!======".format(infos.get('best_val_score', -1),    infos.get('best_epoch', None), infos.get('best_itr', None)))
    print("=================================================================")
    del model
