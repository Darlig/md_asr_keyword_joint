import re
import json
import sys
import yaml
import torchaudio
import torch
import random

import torchaudio.compliance.kaldi as kaldi
import numpy as np
import matplotlib.pyplot as plt

from yamlinclude import YamlIncludeConstructor
from local.utils import  read_list, remove_duplicates_and_blank, compute_cer
from data.loader.data_utils import unfold_list
from model.TransformerKWSPhone import TransformerKWSPhone
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


FBANK_EXTRACTOR = kaldi.fbank
PATTERN = re.compile('^.*?LibriSpeech/')

def plot_result(hyp, ground_truth):

    hyp_np = hyp.numpy()
    ground_truth_np = ground_truth.numpy()

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(ground_truth_np, hyp_np)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, pr_thresholds = precision_recall_curve(ground_truth_np, hyp_np)
    pr_auc = average_precision_score(ground_truth_np, hyp_np)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig('test_double_cross.png', dpi=300)


def sample_keyword(phn_label, n_word=2):

    phn_len = len(phn_label)
    pos = phn_len // 2
    keyword = phn_label[pos:pos+n_word]
    keyword = unfold_list(keyword)
    new_keyword_idx = [i for i in range(len(keyword))]
    md_label = [0 for _ in range(len(keyword))]
    sub_idx = 1

    phn_label = unfold_list(phn_label)
    if len(new_keyword_idx)> 5:
        sub_idx = random.randint(1, len(new_keyword_idx)//5)
    sub_idx = random.sample(new_keyword_idx, k=sub_idx)
    for i in new_keyword_idx:
        if i in sub_idx:
            current_phn = keyword[i]
            sub_phn = random.choice([x for x in range(1, 71) if x != current_phn])
            keyword[i] = sub_phn
            md_label[i] = 1
    keyword_len = torch.tensor([len(keyword)])
    keyword = torch.tensor(keyword, dtype=torch.long)
    md_label = torch.tensor(md_label, dtype=torch.float)
    phn_label = torch.tensor(phn_label, dtype=torch.long)
    return keyword.view(1, -1), keyword_len, md_label.view(1, -1), phn_label.view(1, -1)

def extract_fbank(wav_path, config):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        raise TypeError('wav should be 16k sample rate')
    
    fbank = FBANK_EXTRACTOR(wav, **config)
    fbank_len, dim = fbank.size()
    return fbank.view(1, fbank_len, dim), torch.tensor([fbank_len], dtype=torch.long)

def run(config, ckpt):
    
    ckpt = torch.load(ckpt, map_location='cpu')
    model_state_dict = ckpt['model']

    data_config = config['data_config']
    fbank_config = data_config['sph_config']['feats_config']
    data_list_file = 'train-clean-360/test_100.txt.phn'
    data_list_file = 'test.datalist.phn'
    #data_list_file = 'l2.test.datalist'
    model_config = config['model_config']
    model = TransformerKWSPhone(**model_config)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    
    tr_list = read_list(data_list_file, split_cv=False, shuffle=True)

    hyp = torch.tensor([])
    gd = torch.tensor([])
    e = 0
    t = 0
    for i, one_test_obj in enumerate(tr_list):

        one_test_obj = json.loads(one_test_obj)
        wav_path = PATTERN.sub('', one_test_obj['sph'])
        phn_label = one_test_obj['phn_label']
        #md_label = one_test_obj['md_label']
        #md_label = torch.tensor(md_label)
        #md_label = md_label.view(1,-1)
        #phn_label = unfold_list(phn_label)
        #aug_keyword_len = torch.tensor([len(phn_label)])
        #aug_keyword = torch.tensor([phn_label])

        #phn_label = torch.tensor([phn_label])
        #aug_keyword = aug_keyword.view(1,-1)
        fbank_feats, fbank_len = extract_fbank(wav_path, fbank_config)        
        aug_keyword, aug_keyword_len, md_label, phn_label = sample_keyword(phn_label)
        input_data = (fbank_feats, fbank_len, aug_keyword, aug_keyword_len, md_label) 
        det_result, asr_result = model.evaluate(input_data)
        det_result = det_result.view(-1)
        hyp = torch.cat([hyp, det_result], dim=-1)
        gd = torch.cat([gd, md_label.view(-1)], dim=-1)
        asr_result = remove_duplicates_and_blank(asr_result.view(-1))
        one_error, one_total, one_cer = compute_cer(asr_result, phn_label.view(-1), detail=True)
        e += one_error
        t += one_total
    print (1.0*e/t)
    plot_result(hyp, gd) 
if __name__ == '__main__':

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    config = sys.argv[1]
    ckpt = sys.argv[2]
    config = yaml.load(open(config),Loader=yaml.FullLoader)
    run(config, ckpt)

# 2961-961-0008 BUT THE MEMORY OF THEIR EXPLOITS HAS PASSED AWAY OWING TO THE LAPSE OF TIME AND THE EXTINCTION OF THE ACTORS
#              NOR IS THERE ANY REASON FOR CONSIDERING THE MEMORY FUNCTION AS A PARTICULARLY HIGH OR DIFFICULT PSYCHIC PERFORMANCE IN FACT THE CONTRARY IS TRUE AND YOU CAN FIND A GOOD MEMORY IN PERSONS WHO STAND VERY LOW INTELLECTUALLY