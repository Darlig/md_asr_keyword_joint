#! /usr/bin/env python3

import sys
import json

source_datalist = sys.argv[1]
output_datalist = sys.argv[2]

with open(source_datalist) as f_source, open(output_datalist, 'w') as f_output:
    for line in f_source:
        utt_item = json.loads(line.strip())
        utt_item['sph'] = utt_item['key']
        f_output.write(f"{json.dumps(utt_item)}\n")
