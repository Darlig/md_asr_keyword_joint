#!/usr/bin/env python3

import sys
import json

source_score = sys.argv[1]
output_score = sys.argv[2]

with open(source_score) as f:
    score_item = json.load(f)

for uttid, utt_item in score_item.items():
    for i, word in enumerate(utt_item['words']):
        score_item[uttid]['words'][i].pop('phones_actual')
with open(output_score, 'w') as f:
    json.dump(score_item, f, ensure_ascii=False, indent=4)
