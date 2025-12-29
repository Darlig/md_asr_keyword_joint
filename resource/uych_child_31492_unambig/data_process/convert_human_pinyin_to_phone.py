#!/usr/bin/env python3

import sys
import json
import re

source_human_score = sys.argv[1]
pinyin2phone = sys.argv[2]
output_human_score = sys.argv[3]

def convert_pinyin_to_phone(pinyin_list, pinyin2phone_dict):
    assert len(pinyin_list) % 2 == 0
    pinyin_list = [ [pinyin_list[i], pinyin_list[i + 1] ] for i in range(0, len(pinyin_list), 2) ]
    phone_list = []
    for pinyin in pinyin_list:
        if pinyin[0] == '':
            pinyin[0] = 'none'
        sheng = pinyin[0]
        yun = re.sub('\d$', '', pinyin[1])
        tone = re.sub('[a-z]+', '', pinyin[1])
        phones = pinyin2phone_dict.get(f"{sheng}_{yun}", f"{sheng}_{yun}").split('_')
        phone_list.extend([phones[0], f"{phones[1]}{tone}"])
    return phone_list

with open(pinyin2phone) as f:
    pinyin2phone_dict = json.load(f)

with open(source_human_score) as f_source, open(output_human_score, 'w') as f_output:
    source_item = json.load(f_source)
    for uttid in source_item:
        words = source_item[uttid]['words']
        for i, word in enumerate(words):
            source_item[uttid]['words'][i]['phones'] = convert_pinyin_to_phone(word['phones'], pinyin2phone_dict)
            source_item[uttid]['words'][i]['phones_actual'] = convert_pinyin_to_phone(word['phones_actual'], pinyin2phone_dict)
    json.dump(source_item, f_output, ensure_ascii=False, indent=4)
