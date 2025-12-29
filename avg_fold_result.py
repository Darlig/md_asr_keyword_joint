#! /usr/bin/env python3

#ROC AUC: 0.9234975656715461
#PR AUC: 0.5499680881181014
#@recall=0.1000: precision=0.7788, recall=0.1868, f1_score=0.3014 (max)
#@recall=0.2000: precision=0.7724, recall=0.2017, f1_score=0.3199 (max)
#@recall=0.3000: precision=0.7474, recall=0.3015, f1_score=0.4297 (max)
#@recall=0.4000: precision=0.6985, recall=0.4034, f1_score=0.5114 (max)
#@recall=0.5000: precision=0.6204, recall=0.5032, f1_score=0.5557 (max)
#@recall=0.6000: precision=0.5474, recall=0.6008, f1_score=0.5729 (max)
#@precision=0.1000: precision=0.1000, recall=0.9958, f1_score=0.1817 (max)
#@precision=0.2000: precision=0.2000, recall=0.9130, f1_score=0.3281 (max)
#@precision=0.3000: precision=0.3001, recall=0.8047, f1_score=0.4371 (max)
#@precision=0.4000: precision=0.4002, recall=0.7240, f1_score=0.5155 (max)
#@precision=0.5000: precision=0.5000, recall=0.6263, f1_score=0.5561 (max)
#@precision=0.6000: precision=0.6005, recall=0.5329, f1_score=0.5647 (max)

import sys
import re
import os

fold1_result = sys.argv[1]
output_result = sys.argv[2]

fold_result_pattern = re.sub('fold1', 'fold*', fold1_result)
roc_aucs = []
pr_aucs = []
recall_points = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: []}
precision_points = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: []}
import glob
for fold_result in glob.glob(fold_result_pattern):
    if not os.path.exists(re.sub('\.txt$', '.png', fold_result)):
        continue
    with open(fold_result) as f:
        for line in f:
            if line.startswith('ROC AUC:'):
                roc_aucs.append(float(line.strip().split()[-1]))
            elif line.startswith('PR AUC:'):
                pr_aucs.append(float(line.strip().split()[-1]))
            elif line.startswith('@recall='):
                parts = line.strip().split(':')
                recall_value = float(parts[0].split('=')[1])
                # print(fold_result, parts[1])
                precision = float(re.search(r'precision=([0-9.]+)', parts[1]).group(1))
                # print(parts[1])
                f1_score = float(re.search(r'f1_score=([0-9.]+)', parts[1]).group(1))
                recall_points[recall_value].append([precision, f1_score])
            elif line.startswith('@precision='):
                parts = line.strip().split(':')
                precision_value = float(parts[0].split('=')[1])
                recall = float(re.search(r'recall=([0-9.]+)', parts[1]).group(1))
                f1_score = float(re.search(r'f1_score=([0-9.]+)', parts[1]).group(1))
                precision_points[precision_value].append([recall, f1_score])

os.makedirs(os.path.dirname(output_result), exist_ok=True)
with open(output_result, 'w') as out_f:
    out_f.write(f'ROC AUC: {sum(roc_aucs)/len(roc_aucs)}\n')
    out_f.write(f'PR AUC: {sum(pr_aucs)/len(pr_aucs)}\n')
    for recall_value in sorted(recall_points.keys()):
        avg_precision = sum([point[0] for point in recall_points[recall_value]]) / len(recall_points[recall_value])
        avg_f1 = sum([point[1] for point in recall_points[recall_value]]) / len(recall_points[recall_value])
        out_f.write(f'@recall={recall_value:.4f}: precision={avg_precision:.4f}, f1_score={avg_f1:.4f}\n')
    for precision_value in sorted(precision_points.keys()):
        avg_recall = sum([point[0] for point in precision_points[precision_value]]) / len(precision_points[precision_value])
        avg_f1 = sum([point[1] for point in precision_points[precision_value]]) / len(precision_points[precision_value])
        out_f.write(f'@precision={precision_value:.4f}: recall={avg_recall:.4f}, f1_score={avg_f1:.4f}\n')
