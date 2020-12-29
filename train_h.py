from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import time
import os
from six.moves import cPickle
import traceback
import opts as opts
import models
from dataloader import *
import skimage.io
import eval_utils_h as eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper1 import LossWrapper1
from tqdm import tqdm
import pdb

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

# set the maximum number of CPU kernel
torch.set_num_threads(10)

# write_summary = True
eval_ = True


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


# def add_summary_values(writer, keys, values, iteration):
#     if writer:
#         writer.add_scalar(key, value, iteration)


def train(opt):
    print("=================Training Information==============")
    print("start from {}".format(opt.start_from))
    print("box from {}".format(opt.input_box_dir))
    print("attributes from {}".format(opt.input_att_dir))
    print("features from {}".format(opt.input_fc_dir))
    print("batch size ={}".format(opt.batch_size))
    print("#GPU={}".format(torch.cuda.device_count()))
    print("Caption model {}".format(opt.caption_model))
    print("refine aoa {}".format(opt.refine_aoa))
    print("Number of aoa module {}".format(opt.aoa_num))
    print("Self Critic After  {}".format(opt.self_critical_after))
    print("learning_rate_decay_every {}".format(opt.learning_rate_decay_every))
    
    # use more data to fine tune the model for better challeng results. We dont use it
    if opt.use_val or opt.use_test:  
        print("+++++++++++It is a refining training+++++++++++++++")
        print("===========Val is {} used for training ===========".format('' if opt.use_val else 'not'))
        print("===========Test is {} used for training ===========".format('' if opt.use_test else 'not'))
    print("=====================================================")
    
    # set more detail name of checkpoint paths
    checkpoint_path_suffix = "_bs{}".format(opt.batch_size)
    if opt.use_warmup:
        checkpoint_path_suffix += "_warmup"
    if torch.cuda.device_count() > 1:
        checkpoint_path_suffix += "_gpu{}".format(torch.cuda.device_count())

    if opt.checkpoint_path.endswith('_rl'):
        opt.checkpoint_path = opt.checkpoint_path[:-3] + checkpoint_path_suffix + '_rl'
    else:
        opt.checkpoint_path += checkpoint_path_suffix
    print("Save model to {}".format(opt.checkpoint_path))

    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    if opt.use_box:
        opt.att_feat_size = opt.att_feat_size + 5

    acc_steps = getattr(opt, 'acc_steps', 1)
    name_append = opt.name_append
    if len(name_append) > 0 and name_append[0] != '-':
        name_append = '_' + name_append

    loader = DataLoader(opt)

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.losses_log_every = len(loader.split_ix['train']) // opt.batch_size
    print("Evaluate on each {} iterations".format(opt.losses_log_every))
    if opt.write_summary:
        print("write summary to {}".format(opt.checkpoint_path))
        tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}

    # load  checkpoint
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        infos_path = os.path.join(opt.start_from, 'infos' + name_append + '.pkl')
        print("Load model information {}".format(infos_path))
        with open(infos_path, 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            
            # this sanity check may not work well, and comment it if necessary
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                    "Command line argument and saved model disagree on '%s' " % checkme

        histories_path = os.path.join(opt.start_from, 'histories' + name_append + '.pkl')
        if os.path.isfile(histories_path):
            with open(histories_path, 'rb') as f:
                histories = utils.pickle_load(f)
    else:  # start from scratch
        print("==============================================")
        print("Initialize training process from all begining")
        print("==============================================")
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['vocab'] = loader.get_vocab()
    
    infos['opt'] = opt
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    print("==================start from {} iterations -- {} epoch".format(iteration, epoch))
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    start_Img_idx = loader.iterators['train']
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
        best_epoch = infos.get('best_epoch', None)
        best_cider = infos.get('best_val_score', 0)
        print("========best history val cider score: {} in epoch {}=======".format(best_val_score, best_epoch))

    #  sanity check for the saved model name has a correct index
    if opt.name_append.isdigit() and int(opt.name_append)<100:
        assert int(opt.name_append) - epoch == 1, "dismatch in the model index and the real epoch number"
        epoch += 1
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab
    
    if torch.cuda.device_count()>1:
        dp_model = torch.nn.DataParallel(model)
    else:
        dp_model = model
    lw_model = LossWrapper1(model, opt)  # wrap loss into model
    dp_lw_model = torch.nn.DataParallel(lw_model)

    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'aoa'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer_path = os.path.join(opt.start_from, 'optimizer' + name_append + '.pth')
        if os.path.isfile(optimizer_path):
            print("Loading optimizer............")
            optimizer.load_state_dict(torch.load(optimizer_path))

    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '_' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' % (append))
        torch.save(model.state_dict(), checkpoint_path)
        print("Save model state to {}".format(checkpoint_path))

        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' % (append))
        torch.save(optimizer.state_dict(), optimizer_path)
        print("Save model optimizer to {}".format(optimizer_path))

        with open(os.path.join(opt.checkpoint_path, 'infos' + '%s.pkl' % (append)), 'wb') as f:
            utils.pickle_dump(infos, f)
            print("Save training information to {}".format(
                os.path.join(opt.checkpoint_path, 'infos' + '%s.pkl' % (append))))

        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories' + '%s.pkl' % (append)), 'wb') as f:
                utils.pickle_dump(histories, f)
                print("Save training historyes to {}".format(
                    os.path.join(opt.checkpoint_path, 'histories' + '%s.pkl' % (append))))
    try:
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor * opt.refine_lr_decay
                    else:
                        opt.current_lr = opt.learning_rate
                    infos['current_lr'] = opt.current_lr
                    print("Current Learning Rate is: {}".format(opt.current_lr))
                    utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False

                epoch_done = False
            print("{}th Epoch Training starts now!".format(epoch))
            with tqdm(total=len(loader.split_ix['train']), initial=start_Img_idx) as pbar:
                for i in range(start_Img_idx, len(loader.split_ix['train']), opt.batch_size):
                    start = time.time()
                    if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                        opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                        utils.set_lr(optimizer, opt.current_lr)
                    # Load data from train split (0)
                    data = loader.get_batch('train')
                    # print('Read data:', time.time() - start)

                    if (iteration % acc_steps == 0):
                        optimizer.zero_grad()

                    torch.cuda.synchronize()
                    start = time.time()
                    tmp = [data['fc_feats'], data['att_feats'], data['flag_feats'], data['labels'], data['masks'], data['att_masks']]
                    tmp = [_ if _ is None else _.cuda() for _ in tmp]
                    fc_feats, att_feats, flag_feats, labels, masks, att_masks = tmp

                    model_out = dp_lw_model(fc_feats, att_feats, flag_feats, labels, masks, att_masks, data['gts'],
                                            torch.arange(0, len(data['gts'])), sc_flag)

                    loss = model_out['loss'].mean()
                    loss_sp = loss / acc_steps

                    loss_sp.backward()
                    if (iteration + 1) % acc_steps == 0:
                        utils.clip_gradient(optimizer, opt.grad_clip)
                        optimizer.step()
                    torch.cuda.synchronize()
                    train_loss = loss.item()
                    end = time.time()
                    if not sc_flag:
                        pbar.set_description("iter {:8} (epoch {:2}), train_loss = {:.3f}, time/batch = {:.3f}"
                                             .format(iteration, epoch, train_loss, end - start))
                    else:
                        pbar.set_description("iter {:8} (epoch {:2}), avg_reward = {:.3f}, time/batch = {:.3f}"
                                             .format(iteration, epoch, model_out['reward'].mean(), end - start))

                    # Update the iteration and epoch
                    iteration += 1
                    pbar.update(opt.batch_size)
                    if data['bounds']['wrapped']:
                        epoch += 1
                        epoch_done = True
                        # save after each epoch
                        save_checkpoint(model, infos, optimizer)
                        if epoch > 15:  # To save memory, you can comment this part
                            save_checkpoint(model, infos, optimizer, append=str(epoch))
                        print("=====================================================")
                        print("======Best Cider = {} in epoch {}: iter {}!======".format(best_val_score, best_epoch, infos.get('best_itr', None)))
                        print("=====================================================")

                    # Write training history into summary
                    if (iteration % opt.losses_log_every == 0) and opt.write_summary:
                        # if (iteration % 10== 0) and opt.write_summary:
                        add_summary_value(tb_summary_writer, 'loss/train_loss', train_loss, iteration)
                        if opt.noamopt:
                            opt.current_lr = optimizer.rate()
                        elif opt.reduce_on_plateau:
                            opt.current_lr = optimizer.current_lr
                        add_summary_value(tb_summary_writer, 'hyperparam/learning_rate', opt.current_lr, iteration)
                        add_summary_value(tb_summary_writer, 'hyperparam/scheduled_sampling_prob', model.ss_prob,
                                          iteration)
                        if sc_flag:
                            add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean(), iteration)

                        loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                        lr_history[iteration] = opt.current_lr
                        ss_prob_history[iteration] = model.ss_prob

                    # update infos
                    infos['iter'] = iteration
                    infos['epoch'] = epoch
                    infos['iterators'] = loader.iterators
                    infos['split_ix'] = loader.split_ix

                    # make evaluation on validation set, and save model
                    # unnecessary to eval from the beginning 
                    if (iteration % opt.save_checkpoint_every == 0) and eval_ and epoch > 3:
                        # eval model
                        model_path = os.path.join(opt.checkpoint_path, 'model_itr%s.pth' % (iteration))
                        if opt.use_val and not opt.use_test:
                            val_split = 'test'
                        if not opt.use_val:
                            val_split = 'val'
                        # val_split = 'val'

                        eval_kwargs = {'split': val_split,
                                       'dataset': opt.input_json,
                                       'model': model_path}
                        eval_kwargs.update(vars(opt))
                        val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, lw_model.crit, loader,
                                                                                  eval_kwargs)

                        if opt.reduce_on_plateau:
                            if 'CIDEr' in lang_stats:
                                optimizer.scheduler_step(-lang_stats['CIDEr'])
                            else:
                                optimizer.scheduler_step(val_loss)

                        # Write validation result into summary
                        if opt.write_summary:
                            add_summary_value(tb_summary_writer, 'loss/validation loss', val_loss, iteration)

                            if lang_stats is not None:
                                bleu_dict = {}
                                for k, v in lang_stats.items():
                                    if 'Bleu' in k:
                                        bleu_dict[k] = v
                                if len(bleu_dict) > 0:
                                    tb_summary_writer.add_scalars('val/Bleu', bleu_dict, epoch)

                                for k, v in lang_stats.items():
                                    if 'Bleu' not in k:
                                        add_summary_value(tb_summary_writer, 'val/' + k, v, iteration)
                        val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                         'predictions': predictions}

                        # Save model if is improving on validation result
                        if opt.language_eval == 1:
                            current_score = lang_stats['CIDEr']
                        else:
                            current_score = - val_loss

                        best_flag = False

                        if best_val_score is None or current_score > best_val_score:
                            best_val_score = current_score
                            infos['best_epoch'] = epoch
                            infos['best_itr'] = iteration
                            best_flag = True

                        # Dump miscalleous informations
                        infos['best_val_score'] = best_val_score
                        histories['val_result_history'] = val_result_history
                        histories['loss_history'] = loss_history
                        histories['lr_history'] = lr_history
                        histories['ss_prob_history'] = ss_prob_history

                        save_checkpoint(model, infos, optimizer, histories)
                        if opt.save_history_ckpt:
                            save_checkpoint(model, infos, optimizer, append=str(iteration))

                        if best_flag:
                            best_epoch = epoch
                            save_checkpoint(model, infos, optimizer, append='best')
                            print("update best model at {} iteration--{} epoch".format(iteration, epoch))
                    # reset
                    start_Img_idx = 0
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                print("epoch {} break all".format(epoch))
                save_checkpoint(model, infos, optimizer)
                # save_checkpoint(model, infos, optimizer, append=str(epoch))
                tb_summary_writer.close()
                print("============{} Training Done !==============".format('Refine' if opt.use_test or opt.use_val else ''))
                break
    except (RuntimeError, KeyboardInterrupt):  # KeyboardInterrupt
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer, append='interrupt')
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
print("========================start from {}.".format(opt.start_from))
train(opt)
