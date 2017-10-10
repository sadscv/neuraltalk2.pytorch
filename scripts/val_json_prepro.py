# encoding: utf-8
'''
@author: liujshi
@file: val_json_prepro
@time: 2017/9/13 11:20
@desc: 将验证集的json转成可以测试的数据集
'''

import sys
import json
import hashlib
import jieba

total_dict = {"annotations": [], "images": [],
              "info": {
    "contributor": "He Zheng",
    "description": "CaptionEval",
    "url": "https://github.com/AIChallenger/AI_Challenger.git",
    "version": "1",
    "year": 2017
},
    "licenses": [
    {
        "url": "https://challenger.ai"
    }
],
    "type": "captions"
}

val_data = json.load(open(
    '/home/chen/open/downloads/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'))
id = 1
for i in val_data:
    image_id = i['image_id'][:-4]
    image_hash = int(
        int(hashlib.sha256(image_id).hexdigest(), 16) % sys.maxsize)
    for caption in i['caption']:
        i_cut = jieba.cut(caption, cut_all=False)
        i_cut_contain = ' '.join([word for word in i_cut])
        annotation = {
            "caption": i_cut_contain,
            "id": id,
            "image_id": image_hash}
        image = {"file_name": image_id, "id": image_hash}
        total_dict["annotations"].append(annotation)
        total_dict["images"].append(image)
        id += 1

print(total_dict["annotations"][0])
print(total_dict["images"][0])
json.dump(total_dict, open('val_file.json', 'w'))

import pprint
z = json.load(open('val_file.json'))
pprint.pprint(z["annotations"][:5])
pprint.pprint((z["images"][:5]))
