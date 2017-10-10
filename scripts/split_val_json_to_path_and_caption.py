# -*- coding: utf-8 -*-
#!/usr/bin/env python
# @Time    : 17-10-7 下午10:21
# @Author  : sadscv
# @File    : json2path&caption_segment.py
import json
import os
import re

import jieba



'''
{
  "url":"http://m4.biz.itc.cn/pic/new/n/71/65/Img8296571_n.jpg",
  "image_id":"8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg",
  "caption":[
    "\u4e24\u4e2a\u8863\u7740\u4f11\u95f2\u7684\u4eba\u5728\u5e73\u6574\u7684\u9053\u8def\u4e0a\u4ea4\u8c08",
    "\u4e00\u4e2a\u7a7f\u7740\u7ea2\u8272\u4e0a\u8863\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u7a7f\u7740\u7070\
    \u8272\u88e4\u5b50\u7684\u7537\u4eba\u7ad9\u5728\u5ba4\u5916\u7684\u9053\u8def\u4e0a\u4ea4\u8c08",
    "\u5ba4\u5916\u7684\u516c\u56ed\u91cc\u6709\u4e24\u4e2a\u7a7f\u7740\u957f\u88e4\u7684\u7537\u4eba\u5728\u4ea4\u6d41",
    "\u8857\u9053\u4e0a\u6709\u4e00\u4e2a\u7a7f\u7740\u6df1\u8272\u5916\u5957\u7684\u7537\u4eba\u548c\
    \u4e00\u4e2a\u7a7f\u7740\u7ea2\u8272\u5916\u5957\u7684\u7537\u4eba\u5728\u4ea4\u8c08",
    "\u9053\u8def\u4e0a\u6709\u4e00\u4e2a\u8eab\u7a7f\u7ea2\u8272\u4e0a\u8863\u7684\u7537\u4eba\u5728\
    \u548c\u4e00\u4e2a\u62ac\u7740\u5de6\u624b\u7684\u4eba\u8bb2\u8bdd"
  ]
},
'''
FILE_DIR = '/home/chen/open/challenger/val/'
origin_json_path = FILE_DIR + 'caption_validation_annotations_20170910.json'
CAPTION_FILENAME = 'total_caption.txt'
CAPTION_FILENAME_SEG = 'total_caption.txt.seg'
IMG_ID_FILENAME = 'image_id_file.txt'

if not os.path.isdir(FILE_DIR):
    os.makedirs(FILE_DIR)
def load_json():
    with open(os.path.join(FILE_DIR, CAPTION_FILENAME),'w+') as cap,\
            open(os.path.join(FILE_DIR, IMG_ID_FILENAME), 'w+') as img_id,\
            open(origin_json_path, 'r') as f:
        file = json.load(f)
        count_1 = 0
        count_c = 0
        for i in file:
            count_1 +=1

            # print('count:{}, url:{}'.format(count, i['url']))
            img_id.write(i['image_id']+'\n')
            for c in i['caption']:
                if count_1 == 861:
                    print(c)
                    c = re.sub('[\r\n\t]', '', c)
                    print(c)
                count_c += 1
                c = re.sub('[\r\n\t]', '', c).encode('utf8')
                cap.write(c.strip()+'\n')

        print(count_1)
        print(count_c)


def cut_caption():
    caption_dir = os.path.join(FILE_DIR, CAPTION_FILENAME)
    file_lines = open(caption_dir).readlines()
    count = 0
    with open(os.path.join(FILE_DIR, CAPTION_FILENAME_SEG), 'w+') as caption_seg_dir:
        for line in file_lines:
            if count % 1000 == 0 and count < 149000:
                print(count)
            elif count > 149000:
                print(count)
            count += 1
            caption_cut = jieba.cut(line, cut_all=False)
            caption_cut = ' '.join([word for word in caption_cut])
            # print(caption_cut)
            caption_cut = caption_cut.encode('utf8')
            caption_seg_dir.write(caption_cut)
            # Todo 整理 unicode问题
            # https://pythonhosted.org/kitchen/unicode-frustrations.html#frustration-3-inconsistent-treatment-of-output


def how_many_line(filepath):
    with open(filepath, 'r') as f:
        count = 0
        for line in f:
            count +=1
        print(count)




if __name__ == '__main__':
    load_json()
    cut_caption()
    caption_file_path = os.path.join(FILE_DIR, CAPTION_FILENAME)
    # caption_seg_path = os.path.join(FILE_DIR, CAPTION_FILENAME_SEG)
    how_many_line(caption_file_path)
    # how_many_line(caption_seg_path)