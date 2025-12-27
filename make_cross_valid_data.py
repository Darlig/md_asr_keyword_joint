#! /usr/bin/env pythone

import sys
import random
import json
import os

source_datalist = sys.argv[1]
num_folds = int(sys.argv[2])
output_dir = sys.argv[3]

seed = 42

speakers = list(range(1, 107))
random.seed(seed)
random.shuffle(speakers)
fold_size = len(speakers) // num_folds
folds = [speakers[i*fold_size:(i+1)*fold_size] for i in range(num_folds-1)]
folds.append(speakers[(num_folds-1)*fold_size:])

os.makedirs(output_dir, exist_ok=True)
datalist = []
with open(source_datalist) as f:
    for line in f:
        utt_data = json.loads(line.strip())
        uttid = utt_data["key"]
        speaker_id = int(uttid[1:4])
        datalist.append((speaker_id, utt_data))
for fold_idx in range(num_folds):
    val_speakers = set(folds[fold_idx])
    train_data = [utt_data for spk_id, utt_data in datalist if spk_id not in val_speakers]
    val_data = [utt_data for spk_id, utt_data in datalist if spk_id in val_speakers]
    with open(f"{output_dir}/train_fold{fold_idx+1}.jsonl", "w") as f_train:
        for utt_data in train_data:
            f_train.write(f"{json.dumps(utt_data)}\n")
    with open(f"{output_dir}/val_fold{fold_idx+1}.jsonl", "w") as f_val:
        for utt_data in val_data:
            f_val.write(f"{json.dumps(utt_data)}\n")