#!/usr/bin/env python3

import sys
import json

source_datalist_file = sys.argv[1]
output_datalist_file = sys.argv[2]

with open(source_datalist_file) as f_source, open(output_datalist_file, 'w') as f_output:
    for line in f_source:
        utt_item = json.loads(line.strip())
        utt_sph_item = {
            'key': utt_item['key'],
            'sph': utt_item['sph'],
            'label': utt_item['sph_label'],
            'phn_label': utt_item['sph_label'],
            'kw_candidate': utt_item['kw_candidate']
        }
        f_output.write(f"{json.dumps(utt_sph_item)}\n")
