#!/usr/bin/env python3

import sys
import json
import re
import random

source_datalist = sys.argv[1]
word_misp_map = sys.argv[2]
text_scp = sys.argv[3]
phn2id = sys.argv[4]
output_datalist = sys.argv[5]

negative_candidate_size = 3
min_distance = 1
max_distance_ratio = 0.8
#max_distance_ratio = 0.5

#  "ABBEY": {
#    "canonical_pron": [
#      "AE1",
#      "B",
#      "IY0"
#    ],
#    "neighbors": [
#      {
#        "neighbor": "AB'S",
#        "candidate_pron": [
#          "AE1",
#          "B",
#          "Z"
#        ],
#        "distance": 1,
#        "counts": {
#          "sub": 1,
#          "ins": 0,
#          "del": 0
#        },
#        "ops": [
#          "M",
#          "M",
#          "SUB"
#        ]
#      },
#      {
#        "neighbor": "ABBA",
#        "candidate_pron": [
#          "AE1",
#          "B",
#          "AH0"
#        ],
#        "distance": 1,
#        "counts": {
#          "sub": 1,
#          "ins": 0,
#          "del": 0
#        },
#        "ops": [
#          "M",
#          "M",
#          "SUB"
#        ]
#      },


# load phone2id
phn2id_dict = {}
with open(phn2id) as f_phn2id:
    for line in f_phn2id:
        sym, symid = line.strip().split()
        phn2id_dict[sym] = int(symid)
print(f"phn2id_dict length: {len(phn2id_dict)}")

# load text scp
uttid2text_dict = {}
with open(text_scp) as f_text:
    for line in f_text:
        uttid, text = re.split(r'\s+', line.strip(), maxsplit=1)
        uttid2text_dict[uttid] = text
print(f"uttid2text_dict length: {len(uttid2text_dict)}")

# prepare word mispron map (phone in id, given phone2id)
with open(word_misp_map) as f_word_map:
    word_map_dict = json.load(f_word_map)

word2misp_phn_list = {}
word2cano_phn_list = {}
for word in word_map_dict:
    word2misp_phn_list[word] = []
    word2cano_phn_list[word] = [ phn2id_dict[phn] for phn in word_map_dict[word]["canonical_pron"] ]
    for neighbor in word_map_dict[word]["neighbors"]:
        if neighbor["distance"] >= min_distance and neighbor["ratio"] <= max_distance_ratio:
            misp_phnids = [ phn2id_dict[phn] for phn in neighbor["candidate_pron"] ]
            misp_labels = [ 1 if op != "M" else 0 for op in neighbor["ops"] ]
            word2misp_phn_list[word].append({
                "misp_phnids": misp_phnids,
                "misp_labels": misp_labels
            })
print(f"word2misp_phn_list length: {len(word2misp_phn_list)}")

with open(source_datalist) as f_source_data, open(output_datalist, 'w') as f_output_data:
    for line in f_source_data:
        invalid_negative_candidate = False
        source_utt_data_dict = json.loads(line.strip())
        uttid = source_utt_data_dict["key"]
        phn_label = source_utt_data_dict["phn_label"]
        text = uttid2text_dict[uttid]
        text_seg = text.split()
        #text_seg = re.split(r'[ \']', text)
        if len(phn_label) != len(text_seg):
            #print(f"{uttid} {text} {phn_label}")
            #print(f"length mismatch: text: {len(text_seg)}, phn_label: {len(phn_label)}")
            invalid_negative_candidate = True
        negative_candidate = []
        for i in range(negative_candidate_size):
            if invalid_negative_candidate:
                one_negative_candidate = {
                    "phn_label": phn_label,
                    "md_label": [ [ 0 for phn in word ] for word in phn_label ]
                }
            else:
                one_negative_candidate = {"phn_label": [], "md_label": []}
                for j, word in enumerate(text_seg):
                    if word not in word2misp_phn_list:
                        one_negative_candidate["phn_label"].append(phn_label[j])
                        one_negative_candidate["md_label"].append([ 0 for phn in phn_label[j] ])
                    else:
                        misp_phn_lists = word2misp_phn_list[word]
                        cano_phn_list = word2cano_phn_list[word]
                        # use canonical label if a word has no mispronunciation in word map or the word has multiple canonical pronunciation
                        if misp_phn_lists == [] or cano_phn_list != phn_label[j]:
                            one_negative_candidate["phn_label"].append(phn_label[j])
                            one_negative_candidate["md_label"].append([ 0 for phn in phn_label[j] ])
                        else:
                            one_mispron = random.choice(misp_phn_lists)
                            one_negative_candidate["phn_label"].append(one_mispron["misp_phnids"])
                            one_negative_candidate["md_label"].append(one_mispron["misp_labels"])
            negative_candidate.append(one_negative_candidate)
        out_utt_data_dict = source_utt_data_dict
        out_utt_data_dict.update({"negative_candidate": negative_candidate})
        f_output_data.write(f"{json.dumps(out_utt_data_dict)}\n")
                
