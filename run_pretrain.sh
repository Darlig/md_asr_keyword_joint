#! /bin/bash

L2ARCTIC_ROOT=/work104/weiyang/data/L2-ARCTIC/
LibriSpeech_lexicon=/work104/weiyang/data/LibriSpeech/resource/librispeech-lexicon.txt
librispeech_wav_scp=/work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/train_all_960/wav.scp
librispeech_text=/work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/train_all_960/text

# prepare datalist for librispeech
# 1. download LibriSpeech data and lexicon
# 2. convert audio format to 16k wav, and prepare $librispeech_wav_scp and $librispeech_text
python make_datalist_librispeech.py ${librispeech_wav_scp} ${librispeech_text} ${LibriSpeech_lexicon} phone2id.txt md_data_list/datalist_librispeech_960.txt
shuf md_data_list/datalist_librispeech_960.txt > md_data_list/datalist_librispeech_960_rand.txt
head -n 500 md_data_list/datalist_librispeech_960_rand.txt > md_data_list/datalist_librispeech_960.valid.txt
tail -n 280741 md_data_list/datalist_librispeech_960_rand.txt > md_data_list/datalist_librispeech_960.train.txt

# L2-ARCTIC dataset
# 1. download and unzip L2-ARCTIC at $L2ARCTIC_ROOT
cd $L2ARCTIC_ROOT
for i in $(find */wav/ -name "*.wav"); do target_path="wav_16k/$i"; mkdir -p `dirname $target_path`; sox $i -r 16000 $target_path; done
cd -
find ${L2ARCTIC_ROOT}/wav_16k -name "*.wav" |awk '{print $1,$1}' |sed 's#^[^ ]*/wav_16k/##g' |sed 's/\.wav / /g' |sed 's#/wav/arctic_#_#' > ${L2ARCTIC_ROOT}/wav_16k.scp

# prepare datalist for L2-ARCTIC
python l2arctic_make_human_label_wordseg.py $L2ARCTIC_ROOT phones.txt human_score_l2arctic_test.json
python convert_human_label_json_into_datalist.py human_score_l2arctic_test.json ${L2ARCTIC_ROOT}/wav_16k.scp phone2id.txt md_data_list/datalist.test.l2arctic.txt


# train
bash run_train.sh --config config/pretrain_librispeech_data_aug.yaml


# test
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug/kwatt_asr_0.pt --min_epoch 45 --max_epoch 50
python eval.py config/pretrain_librispeech_data_aug.yaml exp/md_librispeech_data_aug/kwatt_asr_avg_45-50.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug/result l2arctic
