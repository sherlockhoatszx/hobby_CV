# coding:utf8
"""
/data/:
    -idx2word,word2idx
    -noun_idx.pkl
    -train,val,test,idx2img,img2idx
    -img_cmts_dict.pkl imgid:[[cmts1],[cmts2 string]]
    -results.pth
@author: hezhijian
"""
import torch as t
from torch.utils import data
import os
from PIL import Image
import torchvision as tv
import numpy as np

#img_cmts_dic 是{img_id:[[cmt1],[cmt2]]},ix2id,id2ix,word2idx,idxword
#assume all those data all got
#还需要生成[[[cmt1],[cmt2]],[[cmt21],[cmt22]]]类似于这样的东西。
#然后idx 对应 图片名称，也就是地址，idx对应图片的描述，然后从中随机先择一项

def create_collate_fn(padding, eos, max_length=50):
    def collate_fn(img_cap):
        """
        输入dataset的迭代返回，对此进行处理和包装
        将多个样本拼接在一起成一个batch
        输入： list of data，形如
        [(img1, cap1, index1), (img2, cap2, index2) ....]

        拼接策略如下：
        - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
          选取长度最长的句子，将所有句子pad成一样长
        - 长度不够的用</PAD>在结尾PAD
        - 没有START标识符
        - 如果长度刚好和词一样，那么就没有</EOS>

        返回：
        - imgs(Tensor): batch_sie*2048
        - cap_tensor(Tensor): batch_size*max_length
        - lengths(list of int): 长度为batch_size
        - index(list of int): 长度为batch_size
        """
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps, indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths), indexs)

    return collate_fn




class CaptionDataset(data.Dataset):
    '''This is a iterator'''
    def __init__(self,cfg):
        self.cfg = cfg

        #img_dic load
        #word2idx load from pickle

        #captions list of comment load

        self.padding = word2idx.get('end')
        self.end = word2idx.get('end')

        #load idx to image pickle

        self.all_imgs = t.load(cfg.img_feature_path)

    def __getitem__(self,index):

        img = self.all_imgs[index]
        caption = self.captions[index]

        rdn_index = np.random.choice(len(caption),1)[0]
        caption = caption[rdn_index]
        return img,t.LongTensor(caption),index

    def __len__(self):
        return len(self.ix2id)


def get_dataloader(cfg):

    dataset = CaptionDataset(cfg)
    dataloader = data.DataLoader(dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=cfg.shuffle,
                                 num_workers=cfg.num_workers,
                                 collate_fn=create_collate_fn(dataset.padding,dataset.end))

    return dataloader
