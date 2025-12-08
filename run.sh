
L2ARCTIC_ROOT=/work104/weiyang/data/L2-ARCTIC/

# librispeech
python make_datalist_librispeech.py /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/train_all_960/wav.scp /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/train_all_960/text /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/librispeech-lexicon.txt phone2id.txt md_data_list/datalist_librispeech_960.txt
shuf md_data_list/datalist_librispeech_960.txt > md_data_list/datalist_librispeech_960_rand.txt
head -n 500 md_data_list/datalist_librispeech_960_rand.txt > md_data_list/datalist_librispeech_960.valid.txt
tail -n 280741 md_data_list/datalist_librispeech_960_rand.txt > md_data_list/datalist_librispeech_960.train.txt

# l2arctic
cd $L2ARCTIC_ROOT
for i in $(find */wav/ -name "*.wav"); do target_path="wav_16k/$i"; mkdir -p `dirname $target_path`; sox $i -r 16000 $target_path; done
cd -
find ${L2ARCTIC_ROOT}/wav_16k -name "*.wav" |awk '{print $1,$1}' |sed 's#^[^ ]*/wav_16k/##g' |sed 's/\.wav / /g' |sed 's#/wav/arctic_#_#' > ${L2ARCTIC_ROOT}/wav_16k.scp

# prepare datalist for L2-ARCTIC
python l2arctic_make_human_label_wordseg.py $L2ARCTIC_ROOT phones.txt human_score_l2arctic_test.json
python convert_human_label_json_into_datalist.py human_score_l2arctic_test.json wav_16k_l2arctic.scp phone2id.txt md_data_list/datalist.test.l2arctic.txt


# train
bash run_train.sh --config config/pretrain_librispeech_data_aug.yaml


# test
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug/kwatt_asr_0.pt --min_epoch 45 --max_epoch 50
python eval.py config/pretrain_librispeech_data_aug.yaml exp/md_librispeech_data_aug/kwatt_asr_avg_45-50.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug/result l2arctic
