import re
import json
import sys
import yaml
import torchaudio
import torch
import random
import os
import argparse

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

def _recall_at_precision(precision, recall, p, mode="max"):
    """Return (best_recall, matched_precision, f1) at/above target precision p.

    - mode="max": 在 precision >= p 的点集中，找 recall 最大的点；若有并列，取 precision 最小者。
    - mode="interp": 在按 precision 升序插值得到 r=recall(p)，返回 (r, p, f1(p,r))。
    """
    p = float(np.clip(p, 0.0, 1.0))
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)

    if mode == "interp":
        order = np.argsort(precision)
        prec_sorted = precision[order]
        rec_sorted = recall[order]
        r = float(np.interp(p, prec_sorted, rec_sorted))
        f1 = (2 * p * r) / (p + r) if (p + r) > 0.0 else 0.0
        return r, p, f1

    # mode == "max"
    mask = (precision >= p)
    if not np.any(mask):
        return None, None, None  # 达不到目标 precision

    idxs = np.nonzero(mask)[0]
    rec_mask = recall[idxs]
    max_r = np.max(rec_mask)

    # 并列 recall 时取 precision 最小的那个点
    tie_idxs = idxs[rec_mask == max_r]
    best_idx = tie_idxs[np.argmin(precision[tie_idxs])]

    best_r = float(recall[best_idx])
    best_p = float(precision[best_idx])
    f1 = (2 * best_p * best_r) / (best_p + best_r) if (best_p + best_r) > 0.0 else 0.0
    return best_r, best_p, f1

def _precision_at_recall(precision, recall, r, mode="max"):
    """Return (best_precision, matched_recall, f1) at/above target recall r.

    - mode="max": 在 recall >= r 的点集中，找 precision 最大的点；若有并列，取 recall 最大者。
    - mode="interp": 在按 recall 升序插值得到 p=precision(r)，返回 (p, r, f1(p,r))。
    """

    r = float(np.clip(r, 0.0, 1.0))
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)

    if mode == "interp":
        order = np.argsort(recall)
        rec_sorted = recall[order]
        prec_sorted = precision[order]
        p = float(np.interp(r, rec_sorted, prec_sorted))
        f1 = (2 * p * r) / (p + r) if (p + r) > 0.0 else 0.0
        return p, r, f1

    # mode == "max"
    mask = (recall >= r)
    if not np.any(mask):
        return None, None, None  # 达不到目标 recall

    idxs = np.nonzero(mask)[0]
    prec_mask = precision[idxs]
    max_p = np.max(prec_mask)

    # 并列 precision 时取 recall 最大的那个点
    tie_idxs = idxs[prec_mask == max_p]
    best_idx = tie_idxs[np.argmax(recall[tie_idxs])]

    best_p = float(precision[best_idx])
    best_r = float(recall[best_idx])
    f1 = (2 * best_p * best_r) / (best_p + best_r) if (best_p + best_r) > 0.0 else 0.0
    return best_p, best_r, f1


def plot_result(hyp, ground_truth, save_dir, test_id):
    target_precisions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #target_recalls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    mode = 'max'
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
        if target_precisions is not None:
            if isinstance(target_precisions, (float, int)):
                target_precisions = [float(target_precisions)]
            pr_points = []
            #for r in target_recalls:
            #    matched_p, matched_r, matched_f1 = _precision_at_recall(precision, recall, r, mode=mode)
            #    if matched_p is not None:
            #        pr_points.append((r, matched_p))
            #        print(f"@recall={r:.4f}: precision={matched_p:.4f}, recall={matched_r:.4f}, f1_score={matched_f1:.4f} ({mode})")
            #        f_roc.write(f"@recall={r:.4f}: precision={matched_p:.4f}, recall={matched_r:.4f}, f1_score={matched_f1:.4f} ({mode})\n")
            #        #f_roc.write(f"precision@recall={r:.4f} ({mode}): {p:.4f}\n")
            #    else:
            #        print(f"precision@recall={r:.4f} ({mode}): N/A (recall 未达到)")
            #        f_roc.write(f"precision@recall={r:.4f} ({mode}): N/A (recall 未达到)\n")
            for p in target_precisions:
                matched_r, matched_p, matched_f1 = _recall_at_precision(precision, recall, p, mode=mode)
                if matched_r is not None:
                    #pr_points.append((r, matched_p))
                    print(f"@precision={p:.4f}: precision={matched_p:.4f}, recall={matched_r:.4f}, f1_score={matched_f1:.4f} ({mode})")
                    f_roc.write(f"@precision={p:.4f}: precision={matched_p:.4f}, recall={matched_r:.4f}, f1_score={matched_f1:.4f} ({mode})\n")
                    #f_roc.write(f"precision@recall={r:.4f} ({mode}): {p:.4f}\n")
                else:
                    print(f"recall@precision={r:.4f} ({mode}): N/A (recall 未达到)")
                    f_roc.write(f"recall@precision={r:.4f} ({mode}): N/A (recall 未达到)\n")

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

def sample_keyword(phn_label, md_label=None, negative_candidate=None, n_word=2):

    phn_len = len(phn_label)
    pos = phn_len // 2
    keyword = phn_label[pos:pos+n_word]
    phn_label = unfold_list(phn_label)
    if md_label != None:
        #print("given md test set")
        md_label = md_label[pos:pos+n_word]
        md_label = unfold_list(md_label)
    else:
        keyword, md_label = negative_data_aug(keyword, pos, negative_candidate=negative_candidate)
    keyword = unfold_list(keyword)
    md_label = unfold_list(md_label)
    keyword_len = torch.tensor([len(keyword)])
    keyword = torch.tensor(keyword, dtype=torch.long)
    md_label = torch.tensor(md_label, dtype=torch.float)
    phn_label = torch.tensor(phn_label, dtype=torch.long)
    return keyword.view(1, -1), keyword_len, md_label.view(1, -1), phn_label.view(1, -1)

def sample_all_keywords(phn_label, md_label=None, negative_candidate=None, n_word=2):

    phn_len = len(phn_label)
    keywords = []
    keyword_lens = []
    md_labels = []
    for pos in range(0, phn_len, n_word):
        if pos + 2*n_word >= phn_len:
            keyword = phn_label[pos:phn_len]
            if md_label != None:
                #print("given md test set")
                kw_md_label = md_label[pos:phn_len]
                kw_md_label = unfold_list(kw_md_label)
            else:
                keyword, kw_md_label = negative_data_aug(keyword, pos, negative_candidate=negative_candidate)
            keyword = unfold_list(keyword)
            kw_md_label = unfold_list(kw_md_label)
            keyword_len = torch.tensor([len(keyword)])
            keyword = torch.tensor(keyword, dtype=torch.long)
            kw_md_label = torch.tensor(kw_md_label, dtype=torch.float)
            keywords.append(keyword.view(1, -1))
            keyword_lens.append(keyword_len)
            md_labels.append(kw_md_label.view(1, -1))
            break
        else:
            keyword = phn_label[pos:pos+n_word]
            if md_label != None:
                #print("given md test set")
                kw_md_label = md_label[pos:pos+n_word]
                kw_md_label = unfold_list(kw_md_label)
            else:
                keyword, kw_md_label = negative_data_aug(keyword, pos, negative_candidate=negative_candidate)
            keyword = unfold_list(keyword)
            kw_md_label = unfold_list(kw_md_label)
            keyword_len = torch.tensor([len(keyword)])
            keyword = torch.tensor(keyword, dtype=torch.long)
            kw_md_label = torch.tensor(kw_md_label, dtype=torch.float)
            keywords.append(keyword.view(1, -1))
            keyword_lens.append(keyword_len)
            md_labels.append(kw_md_label.view(1, -1))
    phn_label = unfold_list(phn_label)
    phn_label = torch.tensor(phn_label, dtype=torch.long)
    
    return keywords, keyword_lens, md_labels, phn_label.view(1, -1)

def extract_fbank(wav_path, config):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        raise TypeError('wav should be 16k sample rate')
    
    fbank = FBANK_EXTRACTOR(wav, **config)
    fbank_len, dim = fbank.size()
    return fbank.view(1, fbank_len, dim), torch.tensor([fbank_len], dtype=torch.long)

def run(config, ckpt, data_list_file, save_dir, test_id, num_add):
    
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
        aug_keywords, aug_keyword_lens, aug_md_labels, phn_label = sample_all_keywords(phn_label, md_label, negative_candidate, n_word=4)
        #aug_keyword, aug_keyword_len, md_label, phn_label = sample_keyword(phn_label, md_label, negative_candidate)
        for i in range(len(aug_keywords)):
            aug_keyword = aug_keywords[i]
            aug_keyword_len = aug_keyword_lens[i]
            aug_md_label = aug_md_labels[i]
            #print(f"aug_keyword shape: {aug_keyword.shape}, aug_keyword_len shape: {aug_keyword_len.shape}, md_label shape: {md_label.shape}, phn_label shape: {phn_label.shape}")
            #print(f"aug_keyword: {aug_keyword}, aug_keyword_len: {aug_keyword_len}")
            #print(f"md_label: {md_label}")
            #print(f"phn_label: {phn_label}")
            input_data = (fbank_feats, fbank_len, aug_keyword, aug_keyword_len, aug_md_label) 
            det_result, asr_result = model.evaluate(input_data)
            det_result = det_result.view(-1)
            hyp = torch.cat([hyp, det_result], dim=-1)
            gd = torch.cat([gd, aug_md_label.view(-1)], dim=-1)
            asr_result = remove_duplicates_and_blank(asr_result.view(-1))
            one_error, one_total, one_cer = compute_cer(asr_result, phn_label.view(-1), detail=True)
            e += one_error
            t += one_total
    print (1.0*e/t)
    with open(os.path.join(save_dir, f"cer_{test_id}.txt"), 'w') as f_cer:
        f_cer.write(f"CER: {1.0*e/t}\n")
    # add sample to hyp and gd, as if none of addition errors is detected, make recall comparable
    hyp = torch.cat([hyp, torch.zeros(num_add, dtype=hyp.dtype, device=hyp.device)], dim=-1)
    gd = torch.cat([gd, torch.ones(num_add, dtype=gd.dtype, device=gd.device)], dim=-1)
    plot_result(hyp, gd, save_dir, test_id) 
    plot_distribution(hyp, gd, save_dir, test_id) 
    #compute_dcf(gd, hyp, save_dir, test_id, prior_target=0.5)
    compute_dcf(gd, hyp, save_dir, test_id, prior_target=l2arctic_prior)
    #compute_dcf(gd, hyp, save_dir, test_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("data_list_file")
    parser.add_argument("save_dir")
    parser.add_argument("test_id")

    # 新增参数，有默认值
    parser.add_argument("--num-add", type=int, default=0,
                        help="number of addition error which is removed from datalist, for recall punishment")

    args = parser.parse_args()

    config = args.config
    ckpt = args.checkpoint
    data_list_file = args.data_list_file
    save_dir = args.save_dir
    test_id = args.test_id
    num_add = args.num_add

    print("num_add =", num_add)

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    config = yaml.load(open(config),Loader=yaml.FullLoader)
    run(config, ckpt, data_list_file, save_dir, test_id, num_add)

