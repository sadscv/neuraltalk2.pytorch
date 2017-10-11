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
import numpy as np
import jieba
FILE_DIR = '/home/chen/open/challenger/val/'


# 大致就是大写转小写，再构建词汇表。

def build_vocab(params):
    count_thr = params['word_count_threshold']
    # 这儿的txt.seg 文件是每行都代表一个caption.
    caption_seg_dir = os.path.join(FILE_DIR, 'total_caption.txt.seg')
    file_lines = open(caption_seg_dir).readlines()
    # count up the number of words
    counts = {}
    for sent in file_lines:
        for w in sent.strip().split():
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    #print('\n'.join(map(str, cw[:20])))
    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for sent in file_lines:
        txt = sent.strip().split()
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' %(i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent
        # words to
        print('inserting the special UNK token')
        vocab.append('UNK')
    final_captions = []
    for i in file_lines:
        # 示例: new_caption[i] = ['two', 'man', 'playing', 'football', 'in', 'UNK', 'playground']
        new_caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in i.strip().split()]
        final_captions.append(new_caption)
    return vocab, final_captions


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    max_length = 35 #params['max_length']
    N = int(len(imgs)/5)
    M = len(imgs)
    print('M:{}, N:{}'.format(M, N))
    label_arrays = []
    # note: these will be one-indexed

    label_start_ix = np.zeros(N, dtype='uint32')
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')

    L = np.zeros((M, max_length), dtype='uint32')
    for index, line in enumerate(imgs):
        for k, w in enumerate(line):
            if k < max_length:
                L[index, k] = wtoi[w]
        label_length[index] = min(max_length, len(line))
        start_ix = int(index / 5)
        label_start_ix[start_ix] = start_ix * 5
        label_end_ix[start_ix] = start_ix * 5 + 4
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    for i in range(len(label_length)):
        if label_length[i] == 0:
            print('#', i)
    # Todo 重新调查val集有空行的原因。
    # assert np.all(label_length > 0), 'error: some caption had no words?'
    # label_length_array = label_length.tolist()
    # for length in label_length_array:
    #     if length == 0:
    #         print(length)
    print('encoded captions to array of size ', L.shape)
    print(L[:10])
    return L, label_start_ix, label_end_ix, label_length


def main(params):

    seed(123)  # make reproducible

    # create the vocab
    vocab, final_captions = build_vocab(params)
    print('printing final_captions')
    for i in final_captions[:10]:
        print ([w for w in i])
    print('End print final_captions')

    # a 1-indexed vocab translation table
    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(
        final_captions, params, wtoi)

    print(L.shape)
    print(label_start_ix.shape, label_start_ix[:5])
    print(label_end_ix.shape, label_end_ix[:5])
    print(label_length.shape, label_length[:5])

    # create output h5 file
    N = int(len(final_captions)/5)
    f_lb = h5py.File(FILE_DIR + params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # # create output json file
    image_id_lines = open(FILE_DIR + 'image_id_file.txt').readlines()
    print(len(image_id_lines))
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(image_id_lines):

        jimg = {}
        jimg['split'] = 'val'
        jimg['file_path'] = os.path.join('/home/chen/open/challenger/caption_validation_images_20170910/',
                                             img.strip())  # copy it over, might need

            # copy over & mantain an id, if present (e.g. coco ids, useful)
        jimg['id'], _ = img.strip().rsplit('.')

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/home/chen/open/challenger/val/total_caption.txt.seg',
                        help='input json file to process into hdf5')
    parser.add_argument('--output_json', default=FILE_DIR + 'data.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)