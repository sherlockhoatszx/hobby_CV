# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:38:14 2019

@author: hezhijian
"""

# coding:utf8
import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.beam_search import CaptionGenerator
import time


class CaptionModel(nn.Module):
    def __init__(self, opt, word2ix, ix2word):
        super(CaptionModel, self).__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.output_size = len(self.word2ix)
        self.n_layers = opt.n_layers
        self.dropout_p = opt.dropout_p
        self.max_len = opt.max_length
        
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)
        
        self.fc = nn.Linear(2048, opt.rnn_hidden)
        
        self.attn = nn.Linear(self.hidden_size*(2*self.n_layers+1),self.max_length)
        
        self.attn_combine = nn.Linear(self.hidden_size*3,self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.gru = nn.GRU(self.hidden_size,self.hidden_size,
                          bidirectional = True,
                          num_layers = self.n_layers,
                          batch_first = True)
        
        self.out = nn.Linear(self.hidden_size*2,self.output_size)
        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden, len(word2ix))

        # if opt.share_embedding_weights:
        #     # rnn_hidden=embedding_dim的时候才可以
        #     self.embedding.weight

    def forward(self, img_feats, captions, lengths):
        #embeddings shape is batch_size*max_length*embedding_size
        embeddings = self.embedding(captions)
        # img_feats是2048维的向量,通过全连接层转为256维的向量,和词向量一样
        img_feats = self.fc(img_feats).unsqueeze(0)
        # 将img_feats看成第一个词的词向量 
        embeddings = t.cat([img_feats, embeddings], 0)
        # PackedSequence
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        outputs, state = self.rnn(packed_embeddings)
        # lstm的输出作为特征用来分类预测下一个词的序号
        # 因为输入是PackedSequence,所以输出的output也是PackedSequence
        # PackedSequence第一个元素是Variable,第二个元素是batch_sizes,
        # 即batch中每个样本的长度
        pred = self.classifier(outputs[0])
        return pred, state

    def generate(self, img, eos_token='</EOS>',
                 beam_size=3,
                 max_caption_length=30,
                 length_normalization_factor=0.0):
        """
        根据图片生成描述,主要是使用beam search算法以得到更好的描述
        """
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img =img.unsqueeze(0)
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx] for idx in sent])
                     for sent in sentences]
        return sentences

    def states(self):
        opt_state_dict = {attr: getattr(self.opt, attr)
                          for attr in dir(self.opt)
                          if not attr.startswith('__')}
        return {
            'state_dict': self.state_dict(),
            'opt': opt_state_dict
        }

    def save(self, path=None, **kwargs):
        if path is None:
            path = '{prefix}_{time}'.format(prefix=self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        t.save(states, path)
        return path

    def load(self, path, load_opt=False):
        data = t.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self

    def get_optimizer(self, lr):
        return t.optim.Adam(self.parameters(), lr=lr)
