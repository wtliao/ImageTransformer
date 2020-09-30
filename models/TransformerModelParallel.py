# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel_rel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel_rel


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, obj_encoder, rel_encoder, decoder, src_embed, src_rel_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.obj_encoder = obj_encoder
        self.rel_encoder = rel_encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_rel_embed = src_rel_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, src_rel, tgt, src_mask, src_rel_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode_obj(src, src_mask),
                           self.encode_rel(src_rel, src_rel_mask),
                           src_mask, src_rel_mask, tgt, tgt_mask)

    def encode_obj(self, src, src_mask):
        return self.obj_encoder(self.src_embed(src), src_mask)

    def encode_rel(self, src_rel, src_rel_mask):
        return self.rel_encoder(self.src_rel_embed(src_rel), src_rel_mask)

    def decode(self, obj_memory, rel_memory, src_mask, src_rel_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), obj_memory, rel_memory, src_mask, src_rel_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Obj_Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Obj_Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # import ipdb; ipdb.set_trace()
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Rel_Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Rel_Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # import ipdb; ipdb.set_trace()
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, obj_memory, rel_memory, src_mask, src_rel_mask, tgt_mask):
        # import ipdb; ipdb.set_trace()
        for layer in self.layers:
            x = layer(x, obj_memory, rel_memory, src_mask, src_rel_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, src_rel_attn, feed_forward, dropout, fusetype=2):
        """
        fusetype: 1 add, 2 cat along -1
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.src_rel_attn = src_rel_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.fusetype = fusetype
        if fusetype == 2:
            self.reduce_cat = nn.Sequential(nn.Linear(size * 2, size), nn.ReLU())
        elif fusetype == 1:
            self.reduce_cat = lambda x: x

    def forward(self, x, obj_memory, rel_memory, src_mask, src_rel_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        om = obj_memory
        rm = rel_memory
        # import ipdb; ipdb.set_trace()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # FIXME: here maybe wrong!
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, om, om, src_mask)) + \
        #     self.sublayer[2](x, lambda x: self.src_rel_attn(x, rm, rm, src_rel_mask))
        x1 = self.sublayer[1](x, lambda x: self.src_attn(x, om, om, src_mask))
        x2 = self.sublayer[2](x, lambda x: self.src_rel_attn(x, rm, rm, src_rel_mask))
        if self.fusetype == 2:
            x3 = torch.cat((x1, x2), -1)
        # elif self.fusetype == 3:
        #     x3 = torch.cat((x1, x2), 1)
        elif self.fusetype == 1:
            x3 = x1 + x2
        return self.sublayer[3](self.reduce_cat(x3), self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # constant scaling factor
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # seq_pro_img* batch_size

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # FIXME: not good here, just for convinience
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModelParallel(AttModel_rel):

    def make_model(self, src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8,
                   dropout=0.1, fusetype=2):  # N: number of encoder/decoder layer, d_model: dim of entire model, d_ff: dim of model's input, h: number of identical heads of self-attention
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # ffd = PositionwiseFeedForward(d_model*2, d_ff, dropout) # for concatenation
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Obj_Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Rel_Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(attn), c(ff), dropout, fusetype=fusetype), N),
            lambda x: x,
            lambda x: x,
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModelParallel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
            (nn.Linear(self.att_feat_size, self.input_encoding_size),
             nn.ReLU(),
             nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))  # redef att_embed

        delattr(self, 'rel_embed')
        self.rel_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.rel_feat_size),) if self.use_bn else ()) +
            (nn.Linear(self.rel_feat_size, self.input_encoding_size),
             nn.ReLU(),
             nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(0, tgt_vocab,
                                     N=opt.num_layers,
                                     d_model=opt.input_encoding_size,
                                     d_ff=opt.rnn_size, fusetype=self.opt.fuse_type)

    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        obj_memory = self.model.encode_obj(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], obj_memory, att_masks

    def _prepare_rel_feature(self, rel_feats, rel_masks):

        rel_feats, rel_masks = self._prepare_rel_feature_forward(rel_feats, rel_masks)
        rel_memory = self.model.encode_rel(rel_feats, rel_masks)

        return rel_feats[..., :1], rel_memory, rel_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0).int()
            seq_mask[:, 0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _prepare_rel_feature_forward(self, rel_feats, rel_masks=None):
        rel_feats, rel_masks = self.clip_att(rel_feats, rel_masks)

        rel_feats = pack_wrapper(self.rel_embed, rel_feats, rel_masks)

        if rel_masks is None:
            rel_masks = rel_feats.new_ones(rel_feats.shape[:2], dtype=torch.long)
        rel_masks = rel_masks.unsqueeze(-2)

        return rel_feats, rel_masks

    def _forward(self, fc_feats, att_feats, rel_feats, seq, att_masks=None, rel_masks=None):
        # import ipdb; ipdb.set_trace()
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        rel_feats, rel_masks = self._prepare_rel_feature_forward(rel_feats, rel_masks)

        out = self.model(att_feats, rel_feats, seq, att_masks, rel_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, rel_feats_ph, obj_memory, rel_memory, state, att_mask, rel_mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(obj_memory, rel_memory, att_mask, rel_mask,
                                ys,
                                subsequent_mask(ys.size(1))
                                .to(obj_memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
