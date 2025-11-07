#!/usr/bin/env python3

import sys
import json


source_human_json = sys.argv[1]
wav_scp = sys.argv[2]
phone2id = sys.argv[3]
output_datalist = sys.argv[4]

CORR = 0
SUB_NORMAL = -1
SUB_UNK = -2
SUB_DEVI = -3
DEL = -4

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
        utt_phones_accuracy = []
        utt_phones_error_type = []
        for word_human_label in utt_human_label["words"]:
            utt_phones.append([ phn2id_dict[phn] for phn in word_human_label["phones"]])
            utt_phones_accuracy.append([ int(not i) for i in word_human_label["phones-accuracy"]])
            word_phones_error_type = []
            for error_type in word_human_label["phones_error_type"]:
                if error_type == "CORR":
                    word_phones_error_type.append(CORR)
                elif error_type == "SUB_NORMAL":
                    word_phones_error_type.append(SUB_NORMAL)
                elif error_type == "SUB_UNK":
                    word_phones_error_type.append(SUB_UNK)
                elif error_type == "SUB_DEVI":
                    word_phones_error_type.append(SUB_DEVI)
                else:
                    assert error_type == "DEL"
                    word_phones_error_type.append(DEL)
            utt_phones_error_type.append(word_phones_error_type)
            #utt_phones_error_type.append(word_human_label["phones_error_type"])
        utt_datalist = {
            "key": uttid,
            "sph": utt2wav_dict[uttid],
            "phn_label": utt_phones,
            "md_label": utt_phones_accuracy,
            "md_err_type": utt_phones_error_type
        }
        f_datalist.write(f"{json.dumps(utt_datalist)}\n")
