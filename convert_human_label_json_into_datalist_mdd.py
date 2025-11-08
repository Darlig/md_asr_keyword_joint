#!/usr/bin/env python3

import sys
import json


source_human_json = sys.argv[1]
wav_scp = sys.argv[2]
phone2id = sys.argv[3]
output_datalist = sys.argv[4]

utt2wav_dict = {}
with open(wav_scp) as f_wav:
    for line in f_wav:
        uttid, wav_path = line.strip().split()
        utt2wav_dict[uttid] = wav_path

phn2id_dict = {}
with open(phone2id) as f_phn2id:
    for line in f_phn2id:
        phn, phnid = line.strip().split()
        phn2id_dict[phn] = int(phnid)

with open(source_human_json) as f_json, open(output_datalist, 'w') as f_datalist:
    human_label_dict = json.load(f_json)
    for uttid, utt_human_label in human_label_dict.items():
        utt_phones = []
        utt_phones_actual = []
        utt_phones_accuracy = []
        for word_human_label in utt_human_label["words"]:
            utt_phones.append([ phn2id_dict[phn] for phn in word_human_label["phones"]])
            utt_phones_actual.append([ phn2id_dict[phn] for phn in word_human_label["phones_actual"]])
            utt_phones_accuracy.append([ int(not i) for i in word_human_label["phones-accuracy"]])
        utt_datalist = {
            "key": uttid,
            "sph": utt2wav_dict[uttid],
            "sph_label": utt_phones_actual,
            "phn_label": utt_phones,
            "md_label": utt_phones_accuracy,
            "kw_candidate": list(range(len(utt_phones)))
        }
        f_datalist.write(f"{json.dumps(utt_datalist)}\n")
