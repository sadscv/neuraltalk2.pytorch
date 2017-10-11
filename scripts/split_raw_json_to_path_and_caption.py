# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 17-10-7 下午10:21
# @Author  : sadscv
# @File    : json2path&caption_segment.py
import json
import os
import re
import jieba

'''
将原始json 如下格式
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
中的image_id与caption提取出来，对caption做一些处理如.encode('utf8'), re.sub('[\r\n\t])之后,转换为
image_id如下:
8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg
5731c58bfcb4c6638eeb58695dd97c80011770dd.jpg
b18763272c2eb79bfd6f5fb08b2a92973567aad8.jpg
47c6dfe230f7e90c9b67e7fb44275b09cab57506.jpg
ea7e685096301ed46ad582c112f2f96baacafd65.jpg
be44ae5002d9016abb2e315e638509545b5be31d.jpg
c3313738fb776f74331a41cc9980618c50fe0c7e.jpg

total_caption.txt如下：
两个衣着休闲的人在平整的道路上交谈
一个穿着红色上衣的男人和一个穿着灰色裤子的男人站在室外的道路上交谈
室外的公园里有两个穿着长裤的男人在交流
街道上有一个穿着深色外套的男人和一个穿着红色外套的男人在交谈
道路上有一个身穿红色上衣的男人在和一个抬着左手的人讲话

total_caption.txt.seg如下：
两个 衣着 休闲 的 人 在 平整 的 道路 上 交谈
一个 穿着 红色 上衣 的 男人 和 一个 穿着 灰色 裤子 的 男人 站 在 室外 的 道路 上 交谈
室外 的 公园 里 有 两个 穿着 长裤 的 男人 在 交流
街道 上 有 一个 穿着 深色 外套 的 男人 和 一个 穿着 红色 外套 的 男人 在 交谈
道路 上 有 一个 身穿 红色 上衣 的 男人 在 和 一个 抬着 左手 的 人 讲话

'''

origin_json_path = '/home/chen/open/challenger/caption_train_annotations_20170902.json'
FILE_DIR = '/home/chen/open/challenger/'
CAPTION_FILENAME = 'total_caption.txt'
CAPTION_FILENAME_SEG = 'total_caption.txt.seg'
IMG_ID_FILENAME = 'image_id_file.txt'

if not os.path.isdir(FILE_DIR):
    os.makedirs(FILE_DIR)


def load_json():
    with open(os.path.join(FILE_DIR, CAPTION_FILENAME),'w+') as cap,\
            open(os.path.join(FILE_DIR, 'tmp_caption'), 'w+') as cap_2,\
            open(os.path.join(FILE_DIR, IMG_ID_FILENAME), 'w+') as img_id,\
            open(origin_json_path, 'r') as f:
        file = json.load(f)
        count_1 = 0
        count_error = 0
        for i in file:
            count_1 +=1

            # print('count:{}, url:{}'.format(count, i['url']))
            img_id.write(i['image_id']+'\n')
            MOD_FLAG = -1
            # 如果有空行caption,用同样图片的caption复制过来
            ix_c = 0
            for c in i['caption']:
                if len(c) <= 1:
                    print('# mod this')
                    for _ in i['caption']:
                        print(_)
                    MOD_FLAG = ix_c
                ix_c += 1
            if MOD_FLAG == -1:
                for c_0 in i['caption']:
                    c_0 = re.sub('[\r\n\t]', '', c_0).encode('utf8')
                    cap.write(c_0.strip() + '\n')
            else:
                tmp = []
                for c_1 in i['caption']:
                    c_1 = re.sub('[\r\n\t]', '', c_1).encode('utf8')
                    tmp.append(c_1)
                    # print([_.decode('unicode_escape') for _ in tmp])
                if MOD_FLAG < 4:
                    tmp[MOD_FLAG] = tmp[MOD_FLAG + 1]
                else:
                    tmp[MOD_FLAG] = tmp[MOD_FLAG - 1]
                for _ in tmp:
                    cap.write(_.strip() + '\n')
                    cap_2.write(_.strip() + '\n')
                count_error += 1
                ix_c += 1


        print('{} total caption splited'.format(count_1))
        print('{} error fixed'.format(count_error))


def cut_caption():
    caption_dir = os.path.join(FILE_DIR, CAPTION_FILENAME)
    file_lines = open(caption_dir).readlines()
    count = 0
    with open(os.path.join(FILE_DIR, CAPTION_FILENAME_SEG), 'w+') as caption_seg_dir:
        for line in file_lines:
            if count % 1000 == 0 and count < 1000000:
                print(count)
            elif count > 1000000:
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
            count += 1
        print(count)


if __name__ == '__main__':
    load_json()
    cut_caption()
    caption_file_path = os.path.join(FILE_DIR, CAPTION_FILENAME)
    # caption_seg_path = os.path.join(FILE_DIR, CAPTION_FILENAME_SEG)
    how_many_line(caption_file_path)
    # how_many_line(caption_seg_path)
