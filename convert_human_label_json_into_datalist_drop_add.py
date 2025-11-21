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

n_canonical_sil = 0
with open(source_human_json) as f_json, open(output_datalist, 'w') as f_datalist:
    human_label_dict = json.load(f_json)
    for uttid, utt_human_label in human_label_dict.items():
        utt_phones = []
        utt_phones_accuracy = []
        for word_human_label in utt_human_label["words"]:
            utt_phone = []
            for phn in word_human_label["phones"]:
                if phn != 'sil':
                    utt_phone.append(phn2id_dict[phn])
                else:
                    n_canonical_sil += 1
            #utt_phone = [ phn2id_dict[phn] for phn in word_human_label["phones"] if phn != 'sil']
            if utt_phone != []:
                utt_phones.append(utt_phone)
            utt_phone_accuracy = [ int(not acc) for i, acc in enumerate(word_human_label["phones-accuracy"]) if word_human_label["phones"][i] != 'sil']
            if utt_phone_accuracy != []:
                utt_phones_accuracy.append(utt_phone_accuracy)
        utt_datalist = {
            "key": uttid,
            "sph": utt2wav_dict[uttid],
            "phn_label": utt_phones,
            "md_label": utt_phones_accuracy
        }
        f_datalist.write(f"{json.dumps(utt_datalist)}\n")
print(f"num of canonical sil: {n_canonical_sil}")
