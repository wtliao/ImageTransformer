from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch
import pdb
from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .AttModel1 import *
from .AttModel import *
from .TransformerModel import TransformerModel
from .TransformerModel1 import TransformerModel1
from .AoAModel import AoAModel
from .AoAModel0 import AoAModel0
from .AoAModel1 import AoAModel1
from .AoAModel3 import AoAModel3
from .AoAModel3_old import AoAModel3_old
from .AoAModel3_d1 import AoAModel3_d1
from .AoAModel3_d2 import AoAModel3_d2
from .AoAModel3_d3 import AoAModel3_d3
from .AoAModel3_no_c import AoAModel3_no_c
from .AoAModel3_no_p import AoAModel3_no_p
from .AoAModel3_l1 import AoAModel3_l1
from .AoAModel3_l2 import AoAModel3_l2
from .AoAModel3_l3 import AoAModel3_l3
from .AoAModel3_d1_w2 import AoAModel3_d1_w2
from .AoAModel3_d1_w4 import AoAModel3_d1_w4
from .AoAModel3_d1_24heads import AoAModel3_d1_24heads
from .AoAModel4 import AoAModel4
from .AoAModel_relative import AoAModel_relative
from .AoAModel_b import AoAModel_b
from .AoAModel_b_old import AoAModel_b_old


def setup(opt):
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'language_model':
        model = LMModel(opt)
    elif opt.caption_model == 'newfc':
        model = NewFCModel(opt)
    elif opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    # Transformer
    elif opt.caption_model == 'transformer':
        # model = TransformerModel(opt)
        model = TransformerModel(opt)
    # AoANet
    elif opt.caption_model == 'aoa':
        model = AoAModel(opt)
    elif opt.caption_model == 'aoa0':
        model = AoAModel0(opt)
    elif opt.caption_model == 'aoa1':
        model = AoAModel1(opt)
    # elif opt.caption_model == 'aoa2':
    #     model = AoAModel2(opt)
    elif opt.caption_model == 'aoa3':
        model = AoAModel3(opt)
    elif opt.caption_model == 'aoa3_old':
        model = AoAModel3_old(opt)
    elif opt.caption_model == 'aoa3d1':
        model = AoAModel3_d1(opt)
    elif opt.caption_model == 'aoa3d2':
        model = AoAModel3_d2(opt)
    elif opt.caption_model == 'aoa3d3':
        model = AoAModel3_d3(opt)
    elif opt.caption_model == 'aoa3_no_c':
        model = AoAModel3_no_c(opt)
    elif opt.caption_model == 'aoa3_no_p':
        model = AoAModel3_no_p(opt)
    elif opt.caption_model == 'aoa3l1':
        model = AoAModel3_l1(opt)
    elif opt.caption_model == 'aoa3l2':
        model = AoAModel3_l2(opt)
    elif opt.caption_model == 'aoa3l3':
        model = AoAModel3_l3(opt)
    elif opt.caption_model == 'aoa3d1w2':
        model = AoAModel3_d1_w2(opt)
    elif opt.caption_model == 'aoa3d1w4':
        model = AoAModel3_d1_w4(opt)
    elif opt.caption_model == 'aoa3d1_24h':
        model = AoAModel3_d1_24heads(opt)
    elif opt.caption_model == 'aoa4':
        model = AoAModel4(opt)
    elif opt.caption_model == 'aoarelative':
        model = AoAModel_relative(opt)
    elif opt.caption_model == 'aoab':
        model = AoAModel_b(opt)
    elif opt.caption_model == 'aoab_old':
        model = AoAModel_b_old(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        name_append = opt.name_append
        if len(name_append) > 0 and name_append[0] != '-':
            name_append = '_' + name_append
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        print(os.path.join(opt.start_from, "infos" + name_append + ".pkl"))
        assert os.path.isfile(os.path.join(opt.start_from, "infos" + name_append + ".pkl")), "infos.pkl file does not exist in path {}".format(opt.start_from)
        # pdb.set_trace()
        model_name = 'model' + name_append + '.pth'
        print("Loading model {}......".format(model_name))
        model.load_state_dict(torch.load(os.path.join(opt.start_from, model_name)))

    return model
