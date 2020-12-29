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
import misc.utils2 as utils
import ipdb
from tqdm import tqdm

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is', 'are', 'am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'coco-caption/f30k_captions4eval.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if '+' in model_id:  # ensemble
        save_to = 'eval_results/ensemble'
    else:
        save_to = 'eval_results/single'

    if not os.path.isdir(save_to):
        os.mkdir(save_to)
    cache_path = os.path.join(save_to, model_id + '_' + split + '.json')
    # cache_path = os.path.join(save_to, 'tmp_' + split + '.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    # json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...
    with open(cache_path, 'w') as f:
        json.dump(preds_filt, f)
    print("Write prediction results to {}".format(cache_path))

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join(save_to, model_id + '_' + split + '_imgToEval.json')
    # outfile_path = os.path.join(save_to, 'tmp_' + split + '_imgToEval.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval, 'predCaption': preds_filt}, outfile)
    print("Write prediction results to {}".format(outfile_path))
    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose = False
    # Print beam search
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_beam = 0
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    st1 = time.time()

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['flag_feats'], data['labels'], data['masks'],
                   data['att_masks']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            ffc_feats, att_feats, flag_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(ffc_feats, att_feats, flag_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['flag_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, flag_feats, att_masks = tmp

        eval_kwargs['batch_info'] = data['infos']
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            seq = model(fc_feats, att_feats, flag_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
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
        print(n)
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        model_id = eval_kwargs['id']
        if '+' in model_id:
            result_file = model_id
        else:
            model_name = os.path.basename(eval_kwargs['model']).split('.')[0]
            result_file = '_'.join(eval_kwargs['model'].split("/")[-3:-1] + [model_name, model_id])
        lang_stats = language_eval(dataset, predictions, result_file, split)

    end1 = time.time()
    print("Evaluation Done! {}mins {:.1f} seconds".format((end1 - st1) // 60, (end1 - st1) % 60))
    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
