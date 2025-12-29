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

output_score = {}
with open(source_human_score) as f_source, open(output_human_score, 'w') as f_output:
    human_score = json.load(f_source)
    for uttid in human_score:
        if uttid in ids:
            output_score[uttid] = human_score[uttid]
    json.dump(output_score, f_output, ensure_ascii=False, indent=4)
