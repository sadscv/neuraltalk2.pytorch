#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-12 上午12:44
# @Author  : sadscv
# @File    : fusion_train_val_raw_caption.py
import json

import copy

FILE_DIR = '/home/chen/open/challenger/'

train_cap = FILE_DIR + 'caption_train_annotations_20170902.json'
val_cap = FILE_DIR + 'val/caption_validation_annotations_20170910.json'
out_cap = FILE_DIR + 'fusion_train_val_captions.json'

train_cap =  open(train_cap, 'r')
val_cap = open(val_cap, 'r')
out_cap = open(out_cap, 'w+')
train_cap = json.load(train_cap)
val_cap = json.load(val_cap)

out = copy.deepcopy(train_cap)
print(len(out))
for i in val_cap:
    out.append(i)

print(len(out))
json.dump(out, out_cap)