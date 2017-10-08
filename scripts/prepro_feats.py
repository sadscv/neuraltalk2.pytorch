# -*- coding: utf-8 -*-
"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import tqdm as tqdm
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io

from torchvision import transforms as trn

# 把几个transform合起来
preprocess = trn.Compose([
    # trn.ToTensor(),
    # mean: (R, G, B) and std: (R, G, B)
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet


def main(params):
    # 虽然不知道是什么骚操作，但是这里应该返回了一个实例化的resnet model.具体是什么model由params['model']决定。
    net = getattr(resnet, params['model'])()
    # 上一句运行时程序应该自动下载模型，存入相应路径
    # 下面载入对应路径的模型
    net.load_state_dict(torch.load(os.path.join(params['model_root'], params['model'] + '.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    # input_json is compose by lots of unit like below.
    '''
    {
      "url":"http://m4.biz.itc.cn/pic/new/n/71/65/Img8296571_n.jpg",
      "image_id":"8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg",
      "caption":[
        "\u4e24\u4e2a\u8863\u7740\u4f11\u95f2\u7684\u4eba\u5728\u5e73\u6574\u7684\u9053\u8def\u4e0a\u4ea4\u8c08",
        "\u4e00\u4e2a\u7a7f\u7740\u7ea2\u8272\u4e0a\u8863\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u7a7f\u7740\u7070\u8272\u88e4\u5b50\u7684\u7537\u4eba\u7ad9\u5728\u5ba4\u5916\u7684\u9053\u8def\u4e0a\u4ea4\u8c08",
        "\u5ba4\u5916\u7684\u516c\u56ed\u91cc\u6709\u4e24\u4e2a\u7a7f\u7740\u957f\u88e4\u7684\u7537\u4eba\u5728\u4ea4\u6d41",
        "\u8857\u9053\u4e0a\u6709\u4e00\u4e2a\u7a7f\u7740\u6df1\u8272\u5916\u5957\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u7a7f\u7740\u7ea2\u8272\u5916\u5957\u7684\u7537\u4eba\u5728\u4ea4\u8c08",
        "\u9053\u8def\u4e0a\u6709\u4e00\u4e2a\u8eab\u7a7f\u7ea2\u8272\u4e0a\u8863\u7684\u7537\u4eba\u5728\u548c\u4e00\u4e2a\u62ac\u7740\u5de6\u624b\u7684\u4eba\u8bb2\u8bdd"
      ]
    },
    '''
    imgs = json.load(open(params['input_json'], 'r'))
    total_img_ids = []
    for img in imgs['images']:
        total_img_ids.append(img['id'])
    N = len(total_img_ids)

    seed(123)  # make reproducible

    # 以下为图片抽取出的fc,att存放目录
    dir_fc = params['output_dir'] + '_fc'
    dir_att = params['output_dir'] + '_att'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img_id in tqdm.tqdm(enumerate(total_img_ids)):
        # load the image
        # img dir : /home/chen/downloads/ai_challenger_caption_train_20170902
        # img_dir = 'ai_challenger_caption_train_20170902'
        I = skimage.io.imread(os.path.join(params['images_root'],  img_id + '.jpg'))
        # handle grayscale input images,解决灰度图像的问题，一般的图像都是有3个通道，所以是个3d tensor.
        # 但是灰度没有RGB，则是个2D，这儿把2D转成3D
        if len(I.shape) == 2:
            # newaxis:顾名思义，就是在对应的位置添加一个既定的轴，
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        # 转换一下维度，把原来的RGB轴放到第0轴的位置上来。
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        # 这儿的preprocess是把每个RGB通道变得均值相同，再高斯分布归一化。
        I = Variable(preprocess(I), volatile=True)
        # Todo read att, fc at misc/resent_utils.py
        tmp_fc, tmp_att = my_resnet(I, params['att_size'])
        # write to pkl
        np.save(os.path.join(dir_fc, img_id[:-4]), tmp_fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, img_id[:-4]), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, default='data.json', help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='/home/chen/disk/challenger/features/data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='/home/chen/disk/challenger/caption_train_images_20170902',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='/home/chen/codes/neuraltalk2.pytorch/scripts/data/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
