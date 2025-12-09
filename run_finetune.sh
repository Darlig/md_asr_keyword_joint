#! /bin/bash

L2ARCTIC_ROOT=/work104/weiyang/data/L2-ARCTIC/
pretrained_ckpt_dir=/work104/weiyang/project/maolidan_thesis/experiment/md_asr_keyword_joint/exp.dragon05/md_libri960_double_cross_attention_pos0.1_word0.5_vow_con/

# prepare datalist for L2-ARCTIC
python l2arctic_make_human_label_wordseg_mdd.py $L2ARCTIC_ROOT phones.txt human_score_l2arctic_train.json
python convert_human_label_json_into_datalist_mdd.py human_score_l2arctic_train.json ${L2ARCTIC_ROOT}/wav_16k.scp phone2id.txt md_data_list/datalist.train.l2arctic_mdd.txt
python convert_mdd_datalist_to_sph.py md_data_list/datalist.train.l2arctic_mdd.txt md_data_list/datalist.train.l2arctic_sph_label.txt
python l2arctic_make_human_label_wordseg.py $L2ARCTIC_ROOT phones.txt human_score_l2arctic_test.json
python convert_human_label_json_into_datalist.py human_score_l2arctic_test.json ${L2ARCTIC_ROOT}/wav_16k.scp phone2id.txt md_data_list/datalist.test.l2arctic.txt
shuf -n 100 md_data_list/datalist.train.l2arctic_mdd.txt > md_data_list/datalist.train.l2arctic_mdd.samp100.txt
shuf -n 100 md_data_list/datalist.train.l2arctic_sph_label.txt > md_data_list/datalist.train.l2arctic_sph_label.samp100.txt

mkdir -p exp/md_librispeech_data_aug_ft_l2_train_md_only_phone_encoder_only_bce 
cp ${pretrained_ckpt_dir}/kwatt_asr_49.pt exp/md_librispeech_data_aug_ft_l2_train_md_only_phone_encoder_only_bce/
cp ${pretrained_ckpt_dir}/model.yaml exp/md_librispeech_data_aug_ft_l2_train_md_only_phone_encoder_only_bce/
mkdir -p exp/md_librispeech_data_aug_ft_l2_train_md 
cp ${pretrained_ckpt_dir}/kwatt_asr_49.pt exp/md_librispeech_data_aug_ft_l2_train_md/
cp ${pretrained_ckpt_dir}/model.yaml exp/md_librispeech_data_aug_ft_l2_train_md/
mkdir -p exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug_only_phone_encoder_only_bce 
cp ${pretrained_ckpt_dir}/kwatt_asr_49.pt exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug_only_phone_encoder_only_bce/
cp ${pretrained_ckpt_dir}/model.yaml exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug_only_phone_encoder_only_bce/
mkdir -p exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug 
cp ${pretrained_ckpt_dir}/kwatt_asr_49.pt exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug/
cp ${pretrained_ckpt_dir}/model.yaml exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug/
mkdir -p exp/md_librispeech_data_aug_ft_l2_train_sph_only_ctc_freeze_phone 
cp ${pretrained_ckpt_dir}/kwatt_asr_49.pt exp/md_librispeech_data_aug_ft_l2_train_sph_only_ctc_freeze_phone/
cp ${pretrained_ckpt_dir}/model.yaml exp/md_librispeech_data_aug_ft_l2_train_sph_only_ctc_freeze_phone/


# fine-tune
# only fine-tune speech branch (freeze phone branch), with only speech label of L2 data, with only CTC loss
bash run_train_freeze_phone.sh --config config/pretrain_librispeech_ft_l2_sph_only_ctc_freeze_phone.yaml --GPU 0
# only fine-tune phone encoder, with speech label and synthetic MD label of L2 data, with only BCE loss
bash run_train_only_phone_encoder.sh --config config/pretrain_librispeech_ft_l2_sph_md_aug_only_phone_encoder_only_bce.yaml --GPU 0
# fine-tune with speech label and synthetic MD label of L2 data, with both CTC loss and BCE loss
bash run_train.sh --config config/pretrain_librispeech_ft_l2_sph_md_aug.yaml --GPU 0
# only fine-tune phone encoder, with speech label and human MD label of L2 data, with only BCE loss
bash run_train_only_phone_encoder.sh --config config/pretrain_librispeech_ft_l2_md_only_phone_encoder_only_bce.yaml --GPU 0
# fine-tune with speech label and human MD label of L2 data, with both CTC loss and BCE loss
bash run_train.sh --config config/pretrain_librispeech_ft_l2_md.yaml --GPU 0

# test
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug_ft_l2_train_sph_only_ctc_freeze_phone/kwatt_asr_49.pt --min_epoch 80 --max_epoch 90
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug_only_phone_encoder_only_bce/kwatt_asr_49.pt --min_epoch 80 --max_epoch 90
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug/kwatt_asr_49.pt --min_epoch 80 --max_epoch 90
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug_ft_l2_train_md_only_phone_encoder_only_bce/kwatt_asr_49.pt --min_epoch 80 --max_epoch 90
python avg_model_ckpt.py --ckpt exp/md_librispeech_data_aug_ft_l2_train_md/kwatt_asr_49.pt --min_epoch 80 --max_epoch 90

python eval.py config/pretrain_librispeech_ft_l2_sph_only_ctc_freeze_phone.yaml exp/md_librispeech_data_aug_ft_l2_train_sph_only_ctc_freeze_phone/kwatt_asr_avg_80-90.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug_ft_l2_train_sph_only_ctc_freeze_phone/result l2arctic
python eval.py config/pretrain_librispeech_ft_l2_sph_md_aug_only_phone_encoder_only_bce.yaml exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug_only_phone_encoder_only_bce/kwatt_asr_avg_80-90.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug_only_phone_encoder_only_bce/result l2arctic
python eval.py config/pretrain_librispeech_ft_l2_sph_md_aug.yaml exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug/kwatt_asr_avg_80-90.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug_ft_l2_train_sph_md_aug/result l2arctic
python eval.py config/pretrain_librispeech_ft_l2_md_only_phone_encoder_only_bce.yaml exp/md_librispeech_data_aug_ft_l2_train_md_only_phone_encoder_only_bce/kwatt_asr_avg_80-90.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug_ft_l2_train_md_only_phone_encoder_only_bce/result l2arctic
python eval.py config/pretrain_librispeech_ft_l2_md.yaml exp/md_librispeech_data_aug_ft_l2_train_md/kwatt_asr_avg_80-90.pt md_data_list/datalist.test.l2arctic.txt exp/md_librispeech_data_aug_ft_l2_train_md/result l2arctic


