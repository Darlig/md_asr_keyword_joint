#!/usr/bin/env  python3

import sys
import json

source_human_score = sys.argv[1]
output_score = sys.argv[2]

with open(source_human_score) as f_source:
    human_score = json.load(f_source)

output_human_score = {}
for uttid, utt_item in human_score.items():
    ambig_utt = False
    phones = [ phone for word in utt_item['words'] for phone in word['phones_actual'] ]
    for phone in phones:
        if phone == "*":
            #print(phones)
            #exit()
            ambig_utt = True
            break
    if ambig_utt:
        continue
    output_human_score[uttid] = utt_item

with open(output_score, 'w', encoding='utf-8') as f_output:
    json.dump(output_human_score, f_output, ensure_ascii=False, indent=4)
