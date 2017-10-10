#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-10 上午12:01
# @Author  : sadscv
# @File    : split_train_valid_test.py
import json
data_split = open('data_split.json', 'w+')


with open('data.json','r+') as data:
    file = json.load(data)
    out = {}
    out['images'] = []
    for i in range(len(file['images'])):
        tmp_dict = file['images'][i]
        print('count:{}, info:{}'.format(i, tmp_dict))
        if i >= 200000 and i < 205000:
            tmp_dict['split'] = 'val'
            print(i, tmp_dict)
        if i >= 205000:
            tmp_dict['split'] = 'test'
            print(i, tmp_dict)
        out['images'].append(tmp_dict)
    out['ix_to_word'] = file['ix_to_word']
    json.dump(out, data_split)

# with open('data_split.json','r') as data:
#     file = json.load(data)
#     count = 0
#     for i in file['images']:
#         print('count:{}, info:{}'.format(i, 1))