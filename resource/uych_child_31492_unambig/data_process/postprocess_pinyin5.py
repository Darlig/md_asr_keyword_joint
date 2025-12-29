#!/usr/bin/env python3

import sys
import json
import re

source_score = sys.argv[1]
output_score = sys.argv[2]

with open(source_score) as f:
    score_item = json.load(f)

for uttid, utt_item in score_item.items():
    for i, word in enumerate(utt_item['words']):
        phones = word['phones']
        phones_actual = word['phones_actual']
        phones_accuracy = word['phones-accuracy']
        for j in range(0, len(phones), 2):
            if re.findall('\d$', phones[j+1]) == []:
                phones[j+1] = phones[j+1] + '5'
            if re.findall('\d$', phones_actual[j+1]) == []:
                phones_actual[j+1] = phones_actual[j+1] + '5'
            if phones[j+1].endswith('0'):
                phones[j+1] = re.sub('0$', '5', phones[j+1])
            if phones_actual[j+1].endswith('0'):
                phones_actual[j+1] = re.sub('0$', '5', phones_actual[j+1])
            
            #if phones_actual[j] == 'y' and phones_actual[j+1].startswith('ia'):
            #    phones_actual[j+1] = re.sub('^i', '', phones_actual[j+1])
            #if phones_actual[j] == '' and phones_actual[j+1].startswith('iong') and phones[j] == '' and phones[j+1].startswith('ou'):
            #    phones_actual[j] = 'y'
            #    phones_actual[j+1] = re.sub('^i', '', phones_actual[j+1])
            #    phones_accuracy[j] = 0
            #if phones[j] != '':
            #    phones[j+1] = re.sub(r'r(?=\d$)', '', phones[j+1])
        #for j in range(0, len(phones), 2):
        #    if phones[j] == 'l':
        #        phones[j+1] = re.sub('ü', 'v', phones[j+1])
        #for j in range(len(phones)):
        #    phones[j] = re.sub(r'(?<!e)r(?=\d$)', '', phones[j])
        #    #phones[j] = re.sub('ü', 'u', phones[j])
        #for j in range(len(phones)):
        #    if phones_accuracy[j]:
        #        phones_actual[j] = phones[j]
        word['phones'] = phones
        word['phones_actual'] = phones_actual
        word['phones-accuracy'] = phones_accuracy
        score_item[uttid]['words'][i] = word

with open(output_score, 'w') as f:
    json.dump(score_item, f, ensure_ascii=False, indent=4)
