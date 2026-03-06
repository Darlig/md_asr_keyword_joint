#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Make speaker-level cross-validation splits (train/val) and additionally
split a portion of the *train speakers* into a calibration set.

Design goals (per request):
- Keep the original train/val random assignment unchanged (seed=42 shuffle of 106 speakers).
- Only split *within* each fold's train speakers to create calib.
- Deterministic: calib split uses a per-fold RNG derived from the same seed.

Usage:
  python make_cross_valid_data_with_calib.py <source_datalist.jsonl> <num_folds> <output_dir> [calib_ratio]

Args:
  source_datalist.jsonl : jsonl, each line is a dict with key "key" like cXYZ....
  num_folds             : e.g. 10
  output_dir            : directory to write split files
  calib_ratio           : optional float in (0,1). Default 0.1 (10% of train speakers per fold)

Outputs per fold (1-indexed):
  - train_foldK.jsonl  : training set (excluding calib speakers)
  - calib_foldK.jsonl  : calibration set (subset of train speakers)
  - val_foldK.jsonl    : validation set (same as before)

Also writes speaker lists for reproducibility:
  - speakers_val_foldK.txt
  - speakers_calib_foldK.txt
  - speakers_train_foldK.txt
"""

import sys
import random
import json
import os
from typing import List, Tuple

seed = 42
NUM_SPEAKERS = 106  # speakers are 001..106

def parse_args():
    if len(sys.argv) < 4:
        print("Usage: python make_cross_valid_data_with_calib.py <source_datalist> <num_folds> <output_dir> [calib_ratio]", file=sys.stderr)
        sys.exit(1)
    source_datalist = sys.argv[1]
    num_folds = int(sys.argv[2])
    output_dir = sys.argv[3]
    calib_ratio = 0.1
    if len(sys.argv) >= 5:
        calib_ratio = float(sys.argv[4])
    if not (0.0 < calib_ratio < 1.0):
        raise ValueError(f"calib_ratio must be in (0,1), got {calib_ratio}")
    return source_datalist, num_folds, output_dir, calib_ratio

def load_datalist(source_datalist: str) -> List[Tuple[int, dict]]:
    datalist = []
    with open(source_datalist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_data = json.loads(line)
            uttid = utt_data["key"]
            speaker_id = int(uttid[1:4])  # c + 3-digit speaker id
            datalist.append((speaker_id, utt_data))
    return datalist

def write_jsonl(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def write_speakers(path: str, speakers: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in speakers:
            f.write(f"{s:03d}\n")

def main():
    source_datalist, num_folds, output_dir, calib_ratio = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    # 1) Speaker shuffle and fold assignment (UNCHANGED vs your original script)
    speakers = list(range(1, NUM_SPEAKERS + 1))
    random.seed(seed)
    random.shuffle(speakers)

    fold_size = len(speakers) // num_folds
    folds = [speakers[i * fold_size:(i + 1) * fold_size] for i in range(num_folds - 1)]
    folds.append(speakers[(num_folds - 1) * fold_size:])

    # 2) Load utterances
    datalist = load_datalist(source_datalist)

    # 3) For each fold: keep val same; split calib from train speakers
    for fold_idx in range(num_folds):
        fold_id = fold_idx + 1
        val_speakers = set(folds[fold_idx])

        # Train speakers (same definition as before)
        train_speakers_all = [s for s in speakers if s not in val_speakers]

        # Deterministic calib split within train speakers (does NOT affect train/val assignment)
        rng = random.Random(seed + 1000 + fold_idx)
        train_speakers_shuf = train_speakers_all[:]
        rng.shuffle(train_speakers_shuf)

        calib_n = max(1, int(round(len(train_speakers_shuf) * calib_ratio)))
        calib_n = min(calib_n, len(train_speakers_shuf) - 1)  # keep at least 1 speaker in train

        calib_speakers = set(train_speakers_shuf[:calib_n])
        train_speakers = set(train_speakers_shuf[calib_n:])

        train_data = [utt_data for spk_id, utt_data in datalist if spk_id in train_speakers]
        calib_data = [utt_data for spk_id, utt_data in datalist if spk_id in calib_speakers]
        val_data = [utt_data for spk_id, utt_data in datalist if spk_id in val_speakers]

        write_jsonl(os.path.join(output_dir, f"train_fold{fold_id}.jsonl"), train_data)
        write_jsonl(os.path.join(output_dir, f"calib_fold{fold_id}.jsonl"), calib_data)
        write_jsonl(os.path.join(output_dir, f"val_fold{fold_id}.jsonl"), val_data)

        # Optional but very useful for debugging/repro
        write_speakers(os.path.join(output_dir, f"speakers_val_fold{fold_id}.txt"), sorted(val_speakers))
        write_speakers(os.path.join(output_dir, f"speakers_calib_fold{fold_id}.txt"), sorted(calib_speakers))
        write_speakers(os.path.join(output_dir, f"speakers_train_fold{fold_id}.txt"), sorted(train_speakers))

    print(
        f"Done. Wrote train/calib/val splits to: {output_dir}\n"
        f"- num_folds={num_folds}, seed={seed}, num_speakers={NUM_SPEAKERS}, calib_ratio={calib_ratio}"
    )

if __name__ == "__main__":
    main()
