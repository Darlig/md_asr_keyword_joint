#!/usr/bin/env python3

import sys
import json

source_human_score = sys.argv[1]
id_file = sys.argv[2]
output_human_score = sys.argv[3]

ids = []
with open(id_file) as f:
    for line in f:
        id, content = line.strip().split()
        ids.append(id)
ids.sort()

with open(source_human_score) as f_source:
    human_score = json.load(f_source)

output_score = {}
with open(output_human_score, 'w') as f_output:
    for uttid in ids:
        if uttid in human_score:
            output_score[uttid] = human_score[uttid]
    json.dump(output_score, f_output, ensure_ascii=False, indent=4)
