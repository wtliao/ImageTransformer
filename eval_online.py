from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import ipdb
from tqdm import tqdm


def eval_online(model, crit, loader, eval_kwargs={}):
    # print caption
    verbose = eval_kwargs.get('verbose', True)
    verbose = False
    # Print beam search
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_beam = 0
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'online_test') # or  'online_val'
    assert eval_kwargs['input_label_h5'] is None, "online test requires no h5_label_file"
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    predictions = []
    st1 = time.time()

    with torch.no_grad():
        with tqdm(total=len(loader.split_ix[split])) as pbar:
            while True:
                data = loader.get_batch(split)
                n = n + loader.batch_size
                pbar.update(loader.batch_size)

                # forward the model to also get generated samples for each image
                # Only leave one feature for each image, in case duplicate sample
                tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['flag_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
                tmp = [_.cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, flag_feats, att_masks = tmp
                # forward the model to also get generated samples for each image
                with torch.no_grad():
                    seq = model(fc_feats, att_feats, flag_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

                # Print beam search
                if beam_size > 1 and verbose_beam:
                    for i in range(loader.batch_size):
                        print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                        print('--' * 10)
                sents = utils.decode_sequence(loader.get_vocab(), seq)

                for k, sent in enumerate(sents):
                    entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                    # if eval_kwargs.get('dump_path', 0) == 1:
                    #     entry['file_name'] = data['infos'][k]['file_path']
                    predictions.append(entry)
                    if eval_kwargs.get('dump_images', 0) == 1:
                        # dump the raw image to vis/ folder
                        cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
                        print(cmd)
                        os.system(cmd)

                    if verbose:
                        print('image %s: %s' % (entry['image_id'], entry['caption']))

                # if we wrapped around the split or used up val imgs budget then bail
                ix0 = data['bounds']['it_pos_now']
                ix1 = data['bounds']['it_max']
                if num_images != -1:
                    ix1 = min(ix1, num_images)
                for i in range(n - ix1):
                    predictions.pop()

                if verbose:
                    print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

                if data['bounds']['wrapped']:
                    break

                if num_images >= 0 and n >= num_images:
                    break

    #import ipdb; ipdb.set_trace()
    if '+' in eval_kwargs['id']:  # ensemble
        cache_path = os.path.join('online_eval_results', eval_kwargs['id'] + '_' + 'bs' + str(beam_size) + '_' + eval_kwargs['split'] + '.json')
    else:
        model_name = eval_kwargs['model'].split('/')[-2].split('h_')[-1]
        model_id = os.path.basename(eval_kwargs['model']).split('.')[0].split('_')[-1]
        cache_path = os.path.join('online_eval_results', model_name + '_' + 'bs' + str(beam_size) + '_' + model_id + eval_kwargs['split'] +'.json')
    # json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...
    with open(cache_path, 'w') as f:
        json.dump(predictions, f)
    print("Write prediction results to {}".format(cache_path))

    end1 = time.time()
    print("Evaluation Done! {}mins {:.1f} seconds".format((end1 - st1) // 60, (end1 - st1) % 60))

    return predictions
