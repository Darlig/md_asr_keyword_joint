#!/usr/bin/env python3

import sys
import json

source_datalist = sys.argv[1]
phn2id = sys.argv[2]

with open(source_datalist) as f_datalist:
    datalist_lines = f_datalist.readlines()

#{"key": "121-127105-0021", "labels": [[14, 19, 8, 10], [37, 38], [10, 9, 16], [12, 27, 41, 16, 1, 6]], "sph": "/home/shiying/Work/data/librispeech/ASR/test-clean-ssd-wav/121/127105/121-127105-0021.wav", "sym": ["W", "OW1", "N", "T", "Y", "UW1", "T", "EH1", "L", "D", "AH1", "G", "L", "AH0", "S"]}

phn2id_dict = {}
for line in datalist_lines:
    utt_data_dict = json.loads(line.strip())
    phn_ids = [ phn_id for word in utt_data_dict["labels"] for phn_id in word ]
    phns = utt_data_dict["sym"]
    assert len(phn_ids) == len(phns)
    for i, phn in enumerate(phns):
        if phn not in phn2id_dict:
            phn2id_dict[phns[i]] = phn_ids[i]

with open(phn2id, 'w') as f_phn2id:
    for phn, phn_id in phn2id_dict.items():
        f_phn2id.write(f"{phn} {phn_id}\n")
