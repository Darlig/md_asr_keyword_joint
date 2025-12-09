#! /usr/bin/env python3

import sys
import json
import os

wav_scp_file = sys.argv[1]
text_file = sys.argv[2]
lexicon_file = sys.argv[3]
phone2id_file = sys.argv[4]
datalist_file = sys.argv[5]

def load_dict(file_path):
    result = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            key = parts[0]
            if key in result:
                continue
            values = parts[1:]
            result[key] = ' '.join(values)
            if i == 0:
                print(f'Loaded first entry from {file_path}: {key} -> {result[key]}')
    return result

def make_datalist(wav_scp, text, lexicon, phone2id, unk_token='<unk>'):
    datalist = []
    for utt_id in wav_scp:
        if utt_id not in text:
            continue
        words = text[utt_id].split()
        phones = []
        phone_ids = []
        for word in words:
            if word in lexicon:
                phones.append(lexicon[word].split())
            else:
                phones.append([unk_token])
        # assert phones != [], f'No phones found for utterance {utt_id}'
        # assert phones[0] != [], f'First word has no phones for utterance {utt_id}'
        phone_ids = [[int(phone2id.get(phone, phone2id.get(unk_token))) for phone in sublist] for sublist in phones]
        entry = {
            'key': utt_id,
            'sph': wav_scp[utt_id],
            'label': phone_ids,
            'phn_label': phone_ids,
            'kw_candidate': list(range(len(phone_ids)))
        }
        datalist.append(entry)
    return datalist


wav_scp = load_dict(wav_scp_file)

text = load_dict(text_file)
lexicon = load_dict(lexicon_file)
phone2id = load_dict(phone2id_file)

datalist = make_datalist(wav_scp, text, lexicon, phone2id, unk_token='spn')
os.makedirs(os.path.dirname(datalist_file), exist_ok=True)
with open(datalist_file, 'w') as f:
    for entry in datalist:
        f.write(json.dumps(entry) + '\n')
