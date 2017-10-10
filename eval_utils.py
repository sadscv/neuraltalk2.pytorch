# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval(dataset, preds, model_id, split):
    import sys
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    else:
        sys.path.append("f30k-caption")
        annFile = 'f30k-caption/annotations/dataset_flickr30k.json'
    from caption_eval.coco_caption.pycxevalcap.eval import COCOEvalCap
    from caption_eval.coco_caption.pycxtools.coco import COCO

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    # 过滤那些没有出现在验证集中的图片，只保存在其中的。
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

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
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    #out是个dict,maybe分别是metric和其对应的结果
    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    """
    用来计算验证集上的损失
    :param model:
    :param crit:
    :param loader:
    :param eval_kwargs:
    :return:
    """
    verbose = eval_kwargs.get('verbose', True)
    # --num_images: how many images to use when periodically evaluating the loss
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        # 只要split:val 没有warpped,就一直循环，迭代地从loader中用get_batch取data.然后再送入模型评测，获得loss.此时crit函数充当了loss评判标准
        # 但是这还不够。我们还需要抽取图像特征，让模型生成一些句子.将生成的句子与其对应的图片信息（如id,path)等存入predicions列表中。
        # 并对这些句子进行语言评估，评测其流畅性，相关性等,此时需要用到的是language_eval函数 。
        data = loader.get_batch(split)
        # batch_size = opt.batch_size
        n = n + loader.batch_size

        # [[0, 1, 3, 1, 1, 0],
        # [0, 1, 1, 0, 1, 0],
        # [0, 11, 1, 1, 1, 0],
        # [0, 1, 0, 1, 0, 0],
        # [0, 1, 1, 4, 1, 0]]
        # 以上是label_batch样式，shape为 (batch_size * seq_per_img, self.seq_length + 2)
        # 其中内容为一个label的caption. 具体是每个img下的seq_per_img个已经trim了的caption.
        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp
            # 注意这里的loss 是个负数，模型效果越好负得越多
            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        # 这里是从batch_feats中截取属于每张图片的fc_feats,att_feats.
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        # seq 大概是模型根据当前的fc_feats, att_feats生成的caption.
        # Todo read  the model.sample
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        
        #set_trace()
        # seq 代表着一个N×D的矩阵，其中每个元素为ix_to_word中的ix.
        # sents 是一个list,每个元素为一个seq[n]转成的句子str.
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            #entry是每张图片的id以及其对应的预测caption.
            # 这儿的dta['infos']是一个list,其中包含batch_size个元素（代表着一个img的所有信息）
            # 每个元素是一个dict,
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            # 如果opt中带有dump_image，则执行一条命令，把源文件复制到/vis/imgs/img/1,2,3.jpg
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        # it_pos_now it_max 分别代表着batch中的img到了多少张， 共有多少张。
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        # Todo 这里还是不明白 n-ix1
        # 也许是在num_images>it_max的情况下，
        # 迭代到了后面，此时的prediction数量很多。要丢弃掉后面一部分。
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        # Todo read language_eval again.
        #这儿的predictions便是模型预测的captions的集合。
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
