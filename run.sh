
python make_datalist_librispeech.py /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/train_all_960/wav.scp /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/train_all_960/text /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/resource/LibriSpeech/librispeech-lexicon.txt phone2id.txt md_data_list/datalist_librispeech_960.txt
shuf md_data_list/datalist_librispeech_960.txt > md_data_list/datalist_librispeech_960_rand.txt
head -n 500 md_data_list/datalist_librispeech_960_rand.txt > md_data_list/datalist_librispeech_960.valid.txt
tail -n 280741 md_data_list/datalist_librispeech_960_rand.txt > md_data_list/datalist_librispeech_960.train.txt

bash run_train.sh --config config/pretrain_librispeech_data_aug.yaml --GPU 0 --port 22100
