# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

from .AttModel1 import pack_wrapper, AttModel, Attention
from .TransformerModel3_d1_w2 import LayerNorm, attention, attention_d, clones, SublayerConnection, PositionwiseFeedForward
# for decoder


class MultiHeadedDotAttention_d(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention_d, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, x1, x2, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # print(mask.shape)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)
        key1 = x1.narrow(2, 0, 1024)
        value1 = x1.narrow(2, 1024, 1024)
        key2 = x2.narrow(2, 0, 1024)
        value2 = x2.narrow(2, 1024, 1024)
        #key3 = x3.narrow(2, 0, 1024)
        #value3 = x3.narrow(2, 1024, 1024)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key1_ = key1.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value1_ = value1.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key2_ = key2.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value2_ = value2.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            #key3_ = key3.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            #value3_ = value3.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        # print(query.shape)
        # print(key.shape)
        # print(value.shape)
        # print(query_.shape)
        # print(key_.shape)
        # print(value_.shape)
        # print(mask.shape)
        x, self.attn = attention_d(query_, key1_, key2_, value1_, value2_, mask=mask,
                                   dropout=self.dropout)
        # print(x.shape)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #x2 = x2.transpose(1, 2).contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)
        #x3 = x3.transpose(1, 2).contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)
        # print(x1.shape)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x
# for encoder


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v
        print(project_k_v)

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        # query,key(parent node),key(neighbor node),key(child node),value(parent node),value(neighbor node),value(child node)
        #self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)
        self.q_linears = nn.Linear(d_model, d_model * scale)
        self.k_linears = clones(nn.Linear(d_model, d_model * scale), 3)  # (parent,neighbor,child)
        self.v_linears = clones(nn.Linear(d_model, d_model * scale), 3)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, flag, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # print(mask.shape)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_ = self.q_linears(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

            key_p, key_n, key_c = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.k_linears, (key, key, key))]
            value_p, value_n, value_c = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.v_linears, (value, value, value))]

        # Apply attention on all the projected vectors in batch.
        # print(query_.shape)
        # print(key_.shape)
        # print(value_.shape)
        # print(mask.shape)
        x, self.attn = attention(query_, key_p, key_n, key_c, value_p, value_n, value_c, flag, mask=mask,
                                 dropout=self.dropout)
        # print(x.shape)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        # print(x.shape)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, flag, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, flag, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


class AoA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, scale=opt.multi_head_scale, do_aoa=opt.refine_aoa, norm_q=0, dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, getattr(opt, 'aoa_num', 6))  # NOTE: number of aoa module is an inportant hyper-parameter, if not specified, use the most popular 6
        self.norm = LayerNorm(layer.size)

    def forward(self, x, flag, mask):
        for layer in self.layers:
            x = layer(x, flag, mask)
        return self.norm(x)


class AoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'AoA')
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)  # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        if self.decoder_type == 'AoA':
            # AoA layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            # LSTM layer
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            # Base linear layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size), nn.ReLU())

        # if opt.use_multi_head == 1: # TODO, not implemented for now
        #     self.attention = MultiHeadedAddAttention(opt.num_heads, opt.d_model, scale=opt.multi_head_scale)
        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention_d(opt.num_heads, opt.rnn_size, project_k_v=0, scale=opt.multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1)
        else:
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x
        self.implicit1 = nn.Linear(1024, 2048)
        self.implicit2 = nn.Linear(1024, 2048)
        # self.implicit3 = nn.Linear(1024,2048)

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # print(p_att_feats.shape)
        # print(self.multi_head_scale * self.d_model)
        # print(p_att_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model).shape)
        # state[0][1] is the context vector at the last step
        p_att_feats1 = self.implicit1(att_feats)
        p_att_feats2 = self.implicit2(att_feats)
        # p_att_feats3 = self.implicit3(att_feats)
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        # print(self.use_multi_head)

        if self.use_multi_head == 2:
            #att = self.attention(h_att, p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model), p_att_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)
            att = self.attention(h_att, p_att_feats1, p_att_feats2, att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            # save the context vector to state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state


class AoAModel3_d1_w2(AttModel):
    def __init__(self, opt):
        super(AoAModel3_d1_w2, self).__init__(opt)
        self.num_layers = 2
        # mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)
        if opt.use_multi_head == 2:
            del self.ctx2att
            #self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.multi_head_scale * opt.rnn_size)
            self.ctx2att = lambda x: x

        if self.use_mean_feats:
            del self.fc_embed
        if opt.refine:
            self.refiner = AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x, y: x
        self.core = AoA_Decoder_Core(opt)

    def _prepare_feature(self, fc_feats, att_feats, flag_feats, att_masks):
        #import ipdb
        # ipdb.set_trace()
        # print(att_feats.shape)
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        # print(att_feats.shape)

        # embed att feats
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # print(att_feats.shape)
        # print(flag_feats.shape)
        att_feats = self.refiner(att_feats, flag_feats, att_masks)
        # print(att_feats.shape)

        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = torch.mean(att_feats, dim=1)
            else:
                mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)
        # print(p_att_feats.shape)

        return mean_feats, att_feats, p_att_feats, att_masks
