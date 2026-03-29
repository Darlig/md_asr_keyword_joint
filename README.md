# Perception Norm for Mispronunciation Detection

Official implementation of the paper “Perception Norm for Mispronunciation Detection”.

This repository contains code for data preparation, pretraining, fine-tuning, multi-annotator modeling, and evaluation. The main experiment commands are summarized in `run.sh`, with the corresponding configurations under `config/`.

## Overview

The current experimental pipeline includes:

1. Data preparation
2. AISHELL-2 pretraining
3. Fine-tuning

## 1. Data Preparation

First, generate the 10-fold cross validation files for the following datasets:

- `uych_release_sph`
- `uych_release_md`
- `uych_anno20_md`
- `uych_anno21_md`
- `uych_anno30_md`
- `uych_release_test_emb`
- `uych_anno20_test_emb`
- `uych_anno21_test_emb`
- `uych_anno30_test_emb`
- `uych_3anno_test_md_vote_emb_with_calib`

Example commands:

```bash
python make_cross_valid_data.py resource/uych_release/datalist_uych_release_nostar_sph.jsonl 10 uych_release_sph_cv10/
python make_cross_valid_data.py resource/uych_release/datalist_uych_release_test_emb.jsonl 10 uych_release_test_emb_cv10
python make_cross_valid_data_with_calib.py resource/uych_3anno/datalist_uych_3anno_test_md_vote.jsonl 10 uych_3anno_test_md_vote_emb_with_calib_cv10
```

## 2. Pretraining

The pretraining stage uses `config/pretrain_hubert_aishell2_data_aug.yaml` and evaluates on multiple test sets:

- `collective`
- `anno1`
- `anno2`
- `anno3`

Training command:

```bash
bash run_train.sh --config config/pretrain_hubert_aishell2_data_aug.yaml --GPU 0,1 --port 22123
```

Evaluation example:

```bash
python eval_whole_hubert_embed.py \
  config/pretrain_hubert_aishell2_data_aug.yaml \
  exp/md_hubert_aishell2_data_aug/kwatt_asr_25.pt \
  resource/uych_release/datalist_uych_release_test_emb.jsonl \
  exp/md_hubert_aishell2_data_aug/result_ep25 \
  uych_release
```

## 3. Fine-tuning

The fine-tuning stage is organized in a 10-fold cross validation setting. `run.sh` shows the commands for `fold1` only; the same pattern applies to `fold2` through `fold10` by replacing the corresponding configs, datalists, and output directories. The fine-tuning experiments include `Target Norm`, `Perception Norm`, and multi-annotator settings.

### 3.1 Target Norm

The following `fold1` setup is shown as an example:

- training and evaluation for `collective` under `fold1_sph_md_aug`

Training example:

```bash
mkdir -p exp/md_hubert_aishell2_ft_uych_release_train_fold1_sph_md_aug
cp exp/md_hubert_aishell2_data_aug/kwatt_asr_24.pt exp/md_hubert_aishell2_ft_uych_release_train_fold1_sph_md_aug/
bash run_train.sh --config config/pretrain_hubert_aishell2_ft_uych_release_fold1_sph_md_aug.yaml --GPU 0,1 --port 22123
```

Evaluation example:

```bash
python eval_whole_hubert_embed.py \
  config/pretrain_hubert_aishell2_ft_uych_release_fold1_sph_md_aug.yaml \
  exp/md_hubert_aishell2_ft_uych_release_train_fold1_sph_md_aug/kwatt_asr_avg_40-50.pt \
  uych_release_test_emb_cv10/val_fold1.jsonl \
  exp/md_hubert_aishell2_ft_uych_release_train_fold1_sph_md_aug/result_avg40-50 \
  uych_release
```

### 3.2 Perception Norm

The following four `fold1` settings are shown as examples:

- collective: `uych_release_fold1_md`
- anno1: `uych_anno20_fold1_md`
- anno2: `uych_anno21_fold1_md`
- anno3: `uych_anno30_fold1_md`

These experiments are all initialized from the pretrained model `exp/md_hubert_aishell2_data_aug/kwatt_asr_24.pt` and evaluated on the corresponding test sets of each fold.

Training example:

```bash
mkdir -p exp/md_hubert_aishell2_ft_uych_release_train_fold1_md
cp exp/md_hubert_aishell2_data_aug/kwatt_asr_24.pt exp/md_hubert_aishell2_ft_uych_release_train_fold1_md/
bash run_train.sh --config config/pretrain_hubert_aishell2_ft_uych_release_fold1_md.yaml --GPU 0,1 --port 22123
```

Evaluation example:

```bash
python eval_whole_hubert_embed.py \
  config/pretrain_hubert_aishell2_ft_uych_release_fold1_md.yaml \
  exp/md_hubert_aishell2_ft_uych_release_train_fold1_md/kwatt_asr_avg_40-50.pt \
  uych_release_test_emb_cv10/val_fold1.jsonl \
  exp/md_hubert_aishell2_ft_uych_release_train_fold1_md/result_avg40-50 \
  uych_release
```

### 3.3 Multi-Annotator Settings

The multi-annotator settings include score average (temperature-scaled) and a voting model.

As with the other fine-tuning settings, `run.sh` shows `fold1` only, while the full experiment setup should be extended from `fold1` to `fold10`.

#### 3.3.1 Score Average (Temperature-Scaled)

In this setting, each annotator model is first temperature-scaled individually, and the final result is obtained by averaging the calibrated scores. The commands below illustrate the basic calibration and evaluation procedure using the `fold1` `anno1` model as an example:

- `uych_3anno_test_md_vote_emb_with_calib_cv10/calib_fold1.jsonl`
- `uych_3anno_test_md_vote_emb_with_calib_cv10/val_fold1.jsonl`

Example command:

```bash
python fit_temperature.py \
  --config config/pretrain_hubert_aishell2_ft_uych_anno20_fold1_md.yaml \
  --ckpt exp/md_hubert_aishell2_ft_uych_anno20_train_fold1_md/kwatt_asr_avg_40-50.pt \
  --datalist uych_3anno_test_md_vote_emb_with_calib_cv10/calib_fold1.jsonl \
  --out exp/md_hubert_aishell2_ft_uych_anno20_train_fold1_md/temperature_for_3anno_md_vote.pt
```

Evaluation example:

```bash
python eval_whole_hubert_embed_calib2.py \
  --temp_file exp/md_hubert_aishell2_ft_uych_anno20_train_fold1_md/temperature_for_3anno_md_vote.pt \
  config/pretrain_hubert_aishell2_ft_uych_anno20_fold1_md.yaml \
  exp/md_hubert_aishell2_ft_uych_anno20_train_fold1_md/kwatt_asr_avg_40-50.pt \
  uych_3anno_test_md_vote_emb_with_calib_cv10/val_fold1.jsonl \
  exp/md_hubert_aishell2_ft_uych_anno20_train_fold1_md/result_avg40-50/ \
  uych_3anno_test_md_vote_calib_3anno_md_vote
```

The final score average result should be computed by averaging the calibrated scores from all annotator models.

#### 3.3.2 Voting Model

This setting is also organized in 10 folds, with `fold1` shown in `run.sh` as an example. It uses:

- `config/pretrain_hubert_aishell2_ft_uych_3anno_md_vote_fold1_md_only_bce_freeze_speech.yaml`
- `run_train_freeze_speech.sh`

In this setup, the speech branch is frozen to train the voting model, and the model is evaluated on `uych_3anno_test_md_vote_emb_with_calib_cv10/val_fold1.jsonl`.

Example:

```bash
mkdir -p exp/md_hubert_aishell2_ft_uych_3anno_md_vote_train_fold1_md_only_bce_freeze_speech
cp exp/md_hubert_aishell2_data_aug/kwatt_asr_24.pt exp/md_hubert_aishell2_ft_uych_3anno_md_vote_train_fold1_md_only_bce_freeze_speech/
bash run_train_freeze_speech.sh --config config/pretrain_hubert_aishell2_ft_uych_3anno_md_vote_fold1_md_only_bce_freeze_speech.yaml --GPU 0,1 --port 22123
```

Evaluation example:

```bash
python eval_whole_hubert_embed.py \
  config/pretrain_hubert_aishell2_ft_uych_3anno_md_vote_fold1_md_only_bce_freeze_speech.yaml \
  exp/md_hubert_aishell2_ft_uych_3anno_md_vote_train_fold1_md_only_bce_freeze_speech/kwatt_asr_avg_40-50.pt \
  uych_3anno_test_md_vote_emb_with_calib_cv10/val_fold1.jsonl \
  exp/md_hubert_aishell2_ft_uych_3anno_md_vote_train_fold1_md_only_bce_freeze_speech/result_avg40-50 \
  uych_3anno_test_md_vote
```

## Main Scripts

- `run.sh`: experiment command summary
- `run_train.sh`: standard training entry point
- `run_train_freeze_speech.sh`: training entry point with the speech branch frozen
- `eval_whole_hubert_embed.py`: evaluation
- `eval_whole_hubert_embed_calib2.py`: evaluation with calibration
- `fit_temperature.py`: temperature scaling

