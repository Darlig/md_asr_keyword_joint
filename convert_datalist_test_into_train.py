#!/usr/bin/env python3

import sys
import json

test_datalist = sys.argv[1]
train_datalist = sys.argv[2]

with open(test_datalist) as f_test, open(train_datalist, 'w') as f_train:
    for line in f_test:
        utt_test_dict = json.loads(line.strip())
        uttid = utt_test_dict['key']
        audio_path = utt_test_dict['sph']
        phn_label = utt_test_dict['phn_label']
        label = phn_label
        utt_train_dict = {
            "key": utt_test_dict["key"],
            "sph": utt_test_dict["sph"],
            "label": utt_test_dict["phn_label"],
            "phn_label": utt_test_dict["phn_label"],
            "kw_candidate": list(range(len(utt_test_dict["phn_label"])))
        }
        f_train.write(f"{json.dumps(utt_train_dict)}\n")

