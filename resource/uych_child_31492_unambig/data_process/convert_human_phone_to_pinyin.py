#!/usr/bin/env python3

import sys
import json
from pyhanlp import *
import re 

human_json_path = sys.argv[1]
output_pinyin_json_path = sys.argv[2]
convert_map_path = sys.argv[3]
convert_map_reverse_path = sys.argv[4]

def word_to_shengyuntone(word):
    """
    Convert a single word to its Pinyin representation.
    """
    pinyin_list = HanLP.convertToPinyinList(word)
    # for pinyin in pinyin_list:
    #     print("shengmu: {}, yunmu: {}, tone: {}".format(pinyin.getShengmu(), pinyin.getYunmu(), pinyin.getTone()))
    # exit()
    # shengyun_list = [ [str(py.getShengmu()), str(py.getYunmu()), str(py.getPinyinWithoutTone()), str(py.getTone())] for py in pinyin_list ]
    # return [str(p) for p in shengyun_list]
    shengyun_list = []
    for py in pinyin_list:
        py_sheng = str(py.getShengmu())
        py_yun = str(py.getYunmu())
        py_pinyin = str(py.getPinyinWithoutTone())
        py_tone = str(py.getTone())
        py_yun2 = re.sub("^"+py_sheng, "", py_pinyin)  # Remove the sheng from pinyin
        shengyun_list.append([py_sheng, py_yun2, py_tone])
    return shengyun_list

with open(human_json_path, 'r', encoding='utf-8') as f:
    human_data = json.load(f)

output_data = {}
convert_map = {}
n_consonant_miss = 0

for uttid, utt_data in human_data.items():
    text = utt_data['text']
    words = utt_data['words']
    for n, word in enumerate(words):
        word_text = word['text']
        word_phones = word['phones']
        word_phones_accuracy = word['phones-accuracy']
        assert len(word_phones) == len(word_text) * 2
        assert len(word_phones_accuracy) == len(word_text) * 2
        word_phones = [ [word_phones[i], word_phones[i + 1] ] for i in range(0, len(word_phones), 2) ]
        word_phones_accuracy = [ [word_phones_accuracy[i], word_phones_accuracy[i + 1] ] for i in range(0, len(word_phones_accuracy), 2) ]
        word_pinyin = word_to_shengyuntone(word_text)
        # print("word: {}\nphones: {}\npinyin: {}".format(word_text, word_phones, word_pinyin))
        # exit()
        new_word_phones = []
        for i, char_phone in enumerate(word_phones):
            phone_sheng = char_phone[0]
            phone_yun = re.sub(r'\d', '', char_phone[1])  # Remove tone number
            phone_tone = char_phone[1][-1]  # Get the tone number
            pinyin_sheng = word_pinyin[i][0]
            pinyin_yun = word_pinyin[i][1]
            # if "{}_{}".format(phone_sheng, phone_yun) not in convert_map:
                # print(len(word_pinyin[i]))
            phone_shengyun = "_".join([phone_sheng, phone_yun])
            # phone_shengyun_compact = "".join([phone_sheng, phone_yun])
            # pinyin_shengyun = "_".join(word_pinyin[i][:-1])
            pinyin_shengyun = "_".join(word_pinyin[i][:2])  # Use only sheng and yun, ignore tone
            new_word_phones.extend([pinyin_sheng, pinyin_yun + phone_tone])
            if phone_shengyun == pinyin_shengyun:
                continue
            elif phone_shengyun not in convert_map:
                convert_map[phone_shengyun] = [pinyin_shengyun, 1]
            else:
                convert_map[phone_shengyun][1] += 1
            # print(convert_map)
            # exit()
            if pinyin_sheng == 'none':
                n_consonant_miss += 1
        words[n]['phones'] = new_word_phones
            
with open(output_pinyin_json_path, 'w', encoding='utf-8') as f:
    json.dump(human_data, f, ensure_ascii=False, indent=4)
print("len of convert_map: {}".format(len(convert_map)))
convert_map = dict(sorted(convert_map.items(), key=lambda item: item[0]))
with open(convert_map_path, 'w') as f:
    json.dump(convert_map, f, indent=4)
convert_map_reverse = {}
for phone, [pinyin, count] in convert_map.items():
    convert_map_reverse[pinyin] = phone
with open(convert_map_reverse_path, 'w') as f:
    json.dump(convert_map_reverse, f, indent=4)
# print("convert_map: \{")
# for key, value in convert_map.items():
#     print(f'  "{key}": "{value}",')
# print("}")
# print("convert_map: {}".format(convert_map))
print("Number of consonants missing: {}".format(n_consonant_miss))

