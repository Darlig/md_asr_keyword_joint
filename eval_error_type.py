import re
import json
import sys
import yaml
import torchaudio
import torch
import random
import os

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
l2arctic_prior = 0.1377

CORR = 0
SUB_NORMAL = -1
SUB_UNK = -2
SUB_DEVI = -3
DEL = -4

def compute_dcf(y_true, y_scores, save_dir, test_id, cost_miss=1.0, cost_fa=1.0, prior_target=0.5):
    assert cost_miss > 0 and cost_miss <= 1
    assert cost_fa > 0 and cost_fa <= 1
    assert prior_target > 0 and prior_target < 1

    y_scores = y_scores.numpy()
    y_true = y_true.numpy()

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    # skip first row of roc_curve with threshold inf
    fpr, tpr, thresholds_roc = fpr[1:], tpr[1:], thresholds_roc[1:]

    roc_thresh_dict = {}
    for i in range(len(fpr)):
        roc_thresh_dict[thresholds_roc[i]] = (fpr[i], tpr[i])

    # dcf(threshold) = cost_miss * prior_target * p_miss(threshold) + cost_fa * (1 - prior_target) * p_fa(threshold)
    dcf = np.min(cost_miss * prior_target * (1 - tpr) + cost_fa * (1 - prior_target) * fpr)
    dcf_index = np.argmin(cost_miss * prior_target * (1 - tpr) + cost_fa * (1 - prior_target) * fpr)
    dcf_threshold = thresholds_roc[dcf_index]
    pred_pos = y_scores >= dcf_threshold
    TP = int(np.sum(pred_pos & (y_true == 1)))
    FP = int(np.sum(pred_pos & (y_true == 0)))
    P = int(np.sum(y_true == 1))
    dcf_precision = TP / (TP + FP) if (TP + FP) else 0.0
    dcf_recall = TP / P if P else 0.0
    dcf_f1 = 2 * dcf_precision * dcf_recall / (dcf_precision + dcf_recall + 1e-12)
    print("{} DCF:".format(test_id))
    print("cost_miss: {0}, cost_fa: {1}".format(cost_miss, cost_fa))
    print("prior_target: {0}".format(prior_target))
    print("DCF: {0:f}".format(dcf))
    print("DCF threshold: {0}".format(dcf_threshold))
    print("DCF p_miss: {0}".format(1 - tpr[dcf_index]))
    print("DCF p_fa: {0}".format(fpr[dcf_index]))
    print("DCF recall: {0}".format(dcf_recall))
    print("DCF precision: {0}".format(dcf_precision))
    print("DCF F1: {0}".format(dcf_f1))
    with open(os.path.join(save_dir, f"dcf_{test_id}.txt"), 'w') as f_dcf:
        f_dcf.write("cost_miss: {0}, cost_fa: {1}\n".format(cost_miss, cost_fa))
        f_dcf.write("prior_target: {0}\n".format(prior_target))
        f_dcf.write("DCF: {0:f}\n".format(dcf))
        f_dcf.write("DCF threshold: {0}\n".format(dcf_threshold))
        f_dcf.write("DCF p_miss: {0}\n".format(1 - tpr[dcf_index]))
        f_dcf.write("DCF p_fa: {0}\n".format(fpr[dcf_index]))
        f_dcf.write("DCF recall: {0}\n".format(dcf_recall))
        f_dcf.write("DCF precision: {0}\n".format(dcf_precision))
        f_dcf.write("DCF F1: {0}\n".format(dcf_f1))

def plot_result(hyp, ground_truth, save_dir, test_id):

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
    plt.savefig(os.path.join(save_dir, f"roc_pr_{test_id}.png"), dpi=300)
    print(f"ROC AUC: {roc_auc}")
    print(f"PR AUC: {pr_auc}")
    with open(os.path.join(save_dir, f"roc_pr_{test_id}.txt"), 'w') as f_roc:
        f_roc.write(f"ROC AUC: {roc_auc}\n")
        f_roc.write(f"PR AUC: {pr_auc}\n")

def plot_distribution(hyp, ground_truth, save_dir, test_id):
    hyp_np = hyp.numpy()
    ground_truth_np = ground_truth.numpy()

    pos_scores = hyp_np[ground_truth_np == 1]
    neg_scores = hyp_np[ground_truth_np == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    print(f"num of pos: {n_pos}, num of neg: {n_neg}")
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=50, alpha=0.5, label=f"Positive Samples (n={n_pos})", color='g', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.5, label=f"Negative Samples (n={n_neg})", color='r', density=True)
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.title('Score Distribution Density for Positive and Negative Samples')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, f"score_distribution_{test_id}.density.png"), dpi=300)

def negative_data_aug(keyword, pos, negative_candidate=None):
    # random phone substitute
    keyword = unfold_list(keyword)
    new_keyword_idx = [i for i in range(len(keyword))]
    md_label = [0 for _ in range(len(keyword))]
    sub_idx = 1
    if len(new_keyword_idx)> 5:
        sub_idx = random.randint(1, len(new_keyword_idx)//5)
    sub_idx = random.sample(new_keyword_idx, k=sub_idx)
    for i in new_keyword_idx:
        if i in sub_idx:
            current_phn = keyword[i]
            sub_phn = random.choice([x for x in range(1, 71) if x != current_phn])
            keyword[i] = sub_phn
            md_label[i] = 1

    ## negative candidate from word mispronunce map
    #keyword_length = len(keyword)
    #new_keyword_idx = [i for i in range(keyword_length)]
    #md_label = [ [ 0 for phn in word ] for word in keyword ]
    #sub_idx = 1
    #if len(new_keyword_idx)> 3:
    ##if len(new_keyword_idx)> 5:
    #    sub_idx = random.randint(1, len(new_keyword_idx)//2)
    #    #sub_idx = random.randint(1, len(new_keyword_idx)//3)
    #sub_idx = random.sample(new_keyword_idx, k=sub_idx)
    ##print(f"sub_idx: {sub_idx}")
    #assert len(negative_candidate) == 3, "negative_candidate must contain 3 candidates"
    #one_negative_candidate = random.choice(negative_candidate)
    #assert len(one_negative_candidate["phn_label"]) == len(one_negative_candidate["md_label"]), "phn_label and md_label in negative candidate must be in the same length"
    #negative_keyword_candidate = one_negative_candidate["phn_label"][pos: pos+keyword_length]
    #negative_md_candidate = one_negative_candidate["md_label"][pos: pos+keyword_length]
    #dice = random.uniform(0,1)
    #if dice > 0.3:
    #    for i in new_keyword_idx:
    #        if i in sub_idx:
    #            keyword[i] = one_negative_candidate["phn_label"][pos + i]
    #            md_label[i] = one_negative_candidate["md_label"][pos + i]
    
    return keyword, md_label

def sample_keyword(phn_label, md_label=None, error_type=None, negative_candidate=None, n_word=2):

    phn_len = len(phn_label)
    pos = phn_len // 2
    keyword = phn_label[pos:pos+n_word]
    phn_label = unfold_list(phn_label)
    if md_label != None:
        #print("given md test set")
        md_label = md_label[pos:pos+n_word]
        md_label = unfold_list(md_label)
        error_type = error_type[pos:pos+n_word]
        #error_type = unfold_list(error_type)
        error_type = [ t for et in error_type for t in et ]
    else:
        keyword, md_label = negative_data_aug(keyword, pos, negative_candidate=negative_candidate)
    keyword = unfold_list(keyword)
    md_label = unfold_list(md_label)
    #error_type = unfold_list(error_type)
    keyword_len = torch.tensor([len(keyword)])
    keyword = torch.tensor(keyword, dtype=torch.long)
    md_label = torch.tensor(md_label, dtype=torch.float)
    error_type = torch.tensor(error_type, dtype=torch.long)
    phn_label = torch.tensor(phn_label, dtype=torch.long)
    return keyword.view(1, -1), keyword_len, md_label.view(1, -1), error_type.view(1, -1), phn_label.view(1, -1)

def extract_fbank(wav_path, config):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        raise TypeError('wav should be 16k sample rate')
    
    fbank = FBANK_EXTRACTOR(wav, **config)
    fbank_len, dim = fbank.size()
    return fbank.view(1, fbank_len, dim), torch.tensor([fbank_len], dtype=torch.long)

def run(config, ckpt, data_list_file, save_dir, test_id):
    
    os.makedirs(save_dir, exist_ok=True)
    ckpt = torch.load(ckpt, map_location='cpu')
    model_state_dict = ckpt['model']

    data_config = config['data_config']
    fbank_config = data_config['sph_config']['feats_config']
    #data_list_file = 'md_data_list/datalist.test.l2arctic.txt'
    #data_list_file = 'train-clean-360/test_100.txt.phn'
    #data_list_file = 'test.datalist.phn'
    #data_list_file = 'l2.test.datalist'
    model_config = config['model_config']
    model = TransformerKWSPhone(**model_config)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    
    tr_list = read_list(data_list_file, split_cv=False, shuffle=True)

    hyp = torch.tensor([])
    gd = torch.tensor([])
    types = torch.tensor([])
    e = 0
    t = 0
    for i, one_test_obj in enumerate(tr_list):
        if i % 100 == 0:
            print(f"Processing {i}th sample")
        one_test_obj = json.loads(one_test_obj)
        wav_path = one_test_obj['sph']
        #wav_path = PATTERN.sub('', one_test_obj['sph'])
        phn_label = one_test_obj['phn_label']
        md_label = one_test_obj.get('md_label', None)
        error_type = one_test_obj.get('md_err_type', None)
        negative_candidate = one_test_obj.get('negative_candidate', None)
        #assert negative_candidate != None
        #md_label = torch.tensor(md_label)
        #md_label = md_label.view(1,-1)
        #phn_label = unfold_list(phn_label)
        #aug_keyword_len = torch.tensor([len(phn_label)])
        #aug_keyword = torch.tensor([phn_label])

        #phn_label = torch.tensor([phn_label])
        #aug_keyword = aug_keyword.view(1,-1)
        fbank_feats, fbank_len = extract_fbank(wav_path, fbank_config)        
        aug_keyword, aug_keyword_len, md_label, error_type, phn_label = sample_keyword(phn_label, md_label, error_type, negative_candidate)
        #print(f"aug_keyword shape: {aug_keyword.shape}, aug_keyword_len shape: {aug_keyword_len.shape}, md_label shape: {md_label.shape}, phn_label shape: {phn_label.shape}")
        #print(f"aug_keyword: {aug_keyword}, aug_keyword_len: {aug_keyword_len}")
        #print(f"md_label: {md_label}")
        #print(f"phn_label: {phn_label}")
        input_data = (fbank_feats, fbank_len, aug_keyword, aug_keyword_len, md_label) 
        det_result, asr_result = model.evaluate(input_data)
        det_result = det_result.view(-1)
        hyp = torch.cat([hyp, det_result], dim=-1)
        gd = torch.cat([gd, md_label.view(-1)], dim=-1)
        types = torch.cat([types, error_type.view(-1)], dim=-1)
        asr_result = remove_duplicates_and_blank(asr_result.view(-1))
        one_error, one_total, one_cer = compute_cer(asr_result, phn_label.view(-1), detail=True)
        e += one_error
        t += one_total
    print (1.0*e/t)
    with open(os.path.join(save_dir, f"cer_{test_id}.txt"), 'w') as f_cer:
        f_cer.write(f"CER: {1.0*e/t}\n")
    print(f"number of unknown substitute: {hyp[types == SUB_UNK].shape}")
    print(f"number of deviation substitute: {hyp[types == SUB_DEVI].shape}")
    print(f"number of normal substitute: {hyp[types == SUB_NORMAL].shape}")
    print(f"number of delete: {hyp[types == DEL].shape}")
    print("only substitute")
    hyp_sub = hyp[types != DEL]
    gd_sub = gd[types != DEL]
    plot_result(hyp_sub, gd_sub, save_dir, f"{test_id}_sub") 
    plot_distribution(hyp_sub, gd_sub, save_dir, f"{test_id}_sub") 
    compute_dcf(gd_sub, hyp_sub, save_dir, f"{test_id}_sub")

    print("only delete")
    hyp_del = hyp[(types != SUB_NORMAL) & (types != SUB_UNK) & (types != SUB_DEVI)]
    gd_del = gd[(types != SUB_NORMAL) & (types != SUB_UNK) & (types != SUB_DEVI)]
    plot_result(hyp_del, gd_del, save_dir, f"{test_id}_del") 
    plot_distribution(hyp_del, gd_del, save_dir, f"{test_id}_del") 
    compute_dcf(gd_del, hyp_del, save_dir, f"{test_id}_del")

    print("only normal substitute (no unknown, no deviation)")
    hyp_sub_normal = hyp[(types != DEL) & (types != SUB_UNK) & (types != SUB_DEVI)]
    gd_sub_normal = gd[(types != DEL) & (types != SUB_UNK) & (types != SUB_DEVI)]
    plot_result(hyp_sub_normal, gd_sub_normal, save_dir, f"{test_id}_sub_normal") 
    plot_distribution(hyp_sub_normal, gd_sub_normal, save_dir, f"{test_id}_sub_normal") 
    compute_dcf(gd_sub_normal, hyp_sub_normal, save_dir, f"{test_id}_sub_normal")
   
    print("only unknown substitute (no normal, no deviation)")
    hyp_sub_unk = hyp[(types != DEL) & (types != SUB_NORMAL) & (types != SUB_DEVI)]
    gd_sub_unk = gd[(types != DEL) & (types != SUB_NORMAL) & (types != SUB_DEVI)]
    plot_result(hyp_sub_unk, gd_sub_unk, save_dir, f"{test_id}_sub_unk") 
    plot_distribution(hyp_sub_unk, gd_sub_unk, save_dir, f"{test_id}_sub_unk") 
    compute_dcf(gd_sub_unk, hyp_sub_unk, save_dir, f"{test_id}_sub_unk")
   
    print("only deviation substitute (no unknown, no normal)")
    hyp_sub_devi = hyp[(types != DEL) & (types != SUB_UNK) & (types != SUB_NORMAL)]
    gd_sub_devi = gd[(types != DEL) & (types != SUB_UNK) & (types != SUB_NORMAL)]
    plot_result(hyp_sub_devi, gd_sub_devi, save_dir, f"{test_id}_sub_devi") 
    plot_distribution(hyp_sub_devi, gd_sub_devi, save_dir, f"{test_id}_sub_devi") 
    compute_dcf(gd_sub_devi, hyp_sub_devi, save_dir, f"{test_id}_sub_devi")
    
    print("all phone samples")
    plot_result(hyp, gd, save_dir, test_id) 
    plot_distribution(hyp, gd, save_dir, test_id) 
    #compute_dcf(gd, hyp, save_dir, test_id, prior_target=0.5)
    #compute_dcf(gd, hyp, save_dir, test_id, prior_target=l2arctic_prior)
    compute_dcf(gd, hyp, save_dir, test_id)

if __name__ == '__main__':

    if len(sys.argv) != 6:
        print(f"Usage: python {sys.argv[0]} <config> <checkpoint> <data-list-file> <save-dir> <test-id>")
        exit()
    config = sys.argv[1]
    ckpt = sys.argv[2]
    data_list_file = sys.argv[3]
    save_dir = sys.argv[4]
    test_id = sys.argv[5]
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    config = yaml.load(open(config),Loader=yaml.FullLoader)
    run(config, ckpt, data_list_file, save_dir, test_id)

# 2961-961-0008 BUT THE MEMORY OF THEIR EXPLOITS HAS PASSED AWAY OWING TO THE LAPSE OF TIME AND THE EXTINCTION OF THE ACTORS
#              NOR IS THERE ANY REASON FOR CONSIDERING THE MEMORY FUNCTION AS A PARTICULARLY HIGH OR DIFFICULT PSYCHIC PERFORMANCE IN FACT THE CONTRARY IS TRUE AND YOU CAN FIND A GOOD MEMORY IN PERSONS WHO STAND VERY LOW INTELLECTUALLY
