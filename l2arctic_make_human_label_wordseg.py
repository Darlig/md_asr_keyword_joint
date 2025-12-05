#!/usr/bin/env python3

import re
import os
import json
from typing import Dict, List, Tuple, Optional
import sys

def make_human_label_json(source_annot_dir: str, target_json_path: str, phone_list: List[str]):
    # pass
    target_label_dict = {}
    n_invalid_utt = 0
    n_utt = 0
    unk_phn_list = []
    for root, dirs, files in os.walk(source_annot_dir):
        # print(f"Processing {root}")
        if root.split('/')[-1] != 'annotation':
            continue
        subset = root.split('/')[-2]
        #if subset not in ['ABA', 'ASI', 'BWC', 'EBVS', 'ERMS', 'HJK', 'HKK', 'HQTV', 'LXC', 'MBMPS', 'NCC', 'PNV', 'RRBI', 'SKA', 'SVBI', 'THV', 'YBAA', 'YDCK']:
        if subset not in ['NJS', 'TLV', 'TNI', 'TXHC', 'YKWK', 'ZHAA']:
            continue
        print(f"Processing {subset}")
        for file in files:
            if file.endswith('.TextGrid'):
                n_utt += 1
                utt_id = re.sub('.TextGrid', '', file)
                utt_id = re.sub('arctic_', '', utt_id)
                utt_id = f"{subset}_{utt_id}"
                utt_annot_path = os.path.join(root, file)
                annot_dict = utt_annot_to_label(utt_annot_path, phone_list)
                if annot_dict is None:
                    n_invalid_utt += 1
                    continue
                utt_label = annot_dict['label']
                unk_phns = annot_dict['unk_phns']
                for unk_phn in unk_phns:
                    if unk_phn not in unk_phn_list:
                        unk_phn_list.append(unk_phn)
                target_label_dict[utt_id] = utt_label
    print(f"num of invalid utts: {n_invalid_utt}/{n_utt}")
    print(f"num of unk phones: {len(unk_phn_list)}")
    print(f"first 10 unk phones: {unk_phn_list[:10]}")
    with open(target_json_path, 'w', encoding='utf-8') as f:
        json.dump(target_label_dict, f, ensure_ascii=False, indent=4)


def utt_annot_to_label(utt_annot_path: str, phone_list: List[str]) -> Dict[str, any]:
    # 获取带时间戳的word和phone intervals
    word_intervals = get_word_intervals(utt_annot_path)
    phone_intervals = get_phone_intervals(utt_annot_path)
    
    # 过滤掉空的word intervals
    non_empty_word_intervals = get_non_empty_intervals(word_intervals)
    
    # 过滤掉静音和空格的phone intervals
    non_silence_phone_intervals = [interval for interval in phone_intervals 
                                  if interval['text'].strip() not in ['', 'sil', 'sp']]
    
    # 处理phone序列，提取phones和phones_accuracy
    phones = []
    phones_accuracy = []
    unk_phns = []
    
    for phone_interval in non_silence_phone_intervals:
        phn = phone_interval['text']
        if ',' in phn:
            connonical_phn, actual_phn, error_type = re.sub(r'\s+', '', phn).split(',')
            if error_type not in ['s', 'd']:
                # print(f"Error type {error_type} not supported, only support s")
                return None

            connonical_phn = re.sub(r'[`)]', '', connonical_phn).upper()
            if connonical_phn not in phone_list:
                return None
                if connonical_phn not in unk_phns:
                    unk_phns.append(connonical_phn)
            phones.append(connonical_phn)
            phones_accuracy.append(0)
        else:
            phn = re.sub(r'[`)]', '', phn).strip().upper()
            if phn not in phone_list:
                return None
                if phn not in unk_phns:
                    unk_phns.append(phn)
            phones.append(phn)
            phones_accuracy.append(1)
    
    # 根据时间戳将phones按word进行分段
    word_segments = []
    phone_idx = 0
    
    for word_interval in non_empty_word_intervals:
        word_text = word_interval['text']
        word_start = word_interval['xmin']
        word_end = word_interval['xmax']
        
        # 找到属于当前word的phones
        word_phones = []
        word_phones_accuracy = []
        
        # 遍历phones，找到时间戳在word时间范围内的phones
        while phone_idx < len(non_silence_phone_intervals):
            phone_interval = non_silence_phone_intervals[phone_idx]
            phone_start = phone_interval['xmin']
            phone_end = phone_interval['xmax']
            
            # 检查phone是否与当前word有重叠
            if phone_start >= word_end:
                # phone在word之后，停止添加phones
                break
            elif phone_end <= word_start:
                # phone在word之前，跳过
                phone_idx += 1
                continue
            else:
                # phone与word有重叠，添加到当前word
                word_phones.append(phones[phone_idx])
                word_phones_accuracy.append(phones_accuracy[phone_idx])
                phone_idx += 1
        
        # 如果当前word没有对应的phones，尝试找到最接近的phones
        if not word_phones and phone_idx < len(non_silence_phone_intervals):
            # 找到时间戳最接近word的phone
            closest_phone_idx = phone_idx
            min_distance = float('inf')
            
            for i in range(phone_idx, len(non_silence_phone_intervals)):
                phone_interval = non_silence_phone_intervals[i]
                phone_start = phone_interval['xmin']
                phone_end = phone_interval['xmax']
                
                # 计算距离
                if phone_start >= word_end:
                    distance = phone_start - word_end
                elif phone_end <= word_start:
                    distance = word_start - phone_end
                else:
                    distance = 0  # 有重叠
                
                if distance < min_distance:
                    min_distance = distance
                    closest_phone_idx = i
                
                # 如果距离太大，停止搜索
                if distance > 0.1:  # 100ms阈值
                    break
            
            if min_distance < 0.1:  # 100ms阈值
                word_phones.append(phones[closest_phone_idx])
                word_phones_accuracy.append(phones_accuracy[closest_phone_idx])
                phone_idx = closest_phone_idx + 1
        
        word_segments.append({
            'text': word_text,
            'phones': word_phones,
            'phones-accuracy': word_phones_accuracy
        })
    
    # 构建完整的word序列
    annot_word_seq = ' '.join([segment['text'] for segment in word_segments])
    
    return {
        'label': {
            'text': annot_word_seq,
            'words': word_segments
        },
        'unk_phns': list(unk_phns)
    }


def parse_textgrid_file(file_path: str) -> Dict[str, List[Dict[str, any]]]:
    """
    解析TextGrid文件，提取每个tier中所有intervals的文本内容
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        字典，键为tier名称，值为包含interval信息的列表
        每个interval包含: xmin, xmax, text
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取总的时间范围
    xmin_match = re.search(r'xmin = ([\d.]+)', content)
    xmax_match = re.search(r'xmax = ([\d.]+)', content)
    
    if not xmin_match or not xmax_match:
        raise ValueError("无法解析TextGrid文件的时间范围")
    
    total_xmin = float(xmin_match.group(1))
    total_xmax = float(xmax_match.group(1))
    
    # 提取tier数量
    size_match = re.search(r'size = (\d+)', content)
    if not size_match:
        raise ValueError("无法解析TextGrid文件的tier数量")
    
    tier_count = int(size_match.group(1))
    
    result = {}
    
    # 解析每个tier
    for i in range(1, tier_count + 1):
        tier_pattern = rf'item \[{i}\]:\s*class = "([^"]+)"\s*name = "([^"]+)"\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*intervals: size = (\d+)\s*(.*?)(?=item \[{i+1}\]|$)'
        tier_match = re.search(tier_pattern, content, re.DOTALL)
        
        if not tier_match:
            continue
            
        tier_class = tier_match.group(1)
        tier_name = tier_match.group(2)
        tier_xmin = float(tier_match.group(3))
        tier_xmax = float(tier_match.group(4))
        intervals_size = int(tier_match.group(5))
        intervals_content = tier_match.group(6)
        
        # 解析intervals
        intervals = []
        for j in range(1, intervals_size + 1):
            interval_pattern = rf'intervals \[{j}\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
            interval_match = re.search(interval_pattern, intervals_content)
            
            if interval_match:
                interval_xmin = float(interval_match.group(1))
                interval_xmax = float(interval_match.group(2))
                interval_text = interval_match.group(3)
                
                intervals.append({
                    'xmin': interval_xmin,
                    'xmax': interval_xmax,
                    'text': interval_text
                })
        
        result[tier_name] = intervals
    
    return result


def get_phone_intervals(file_path: str) -> List[Dict[str, any]]:
    """
    专门提取phones tier中的所有intervals
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        包含所有phone intervals的列表
    """
    textgrid_data = parse_textgrid_file(file_path)
    
    if 'phones' not in textgrid_data:
        raise ValueError("TextGrid文件中未找到'phones' tier")
    
    return textgrid_data['phones']


def get_word_intervals(file_path: str) -> List[Dict[str, any]]:
    """
    专门提取words tier中的所有intervals
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        包含所有word intervals的列表
    """
    textgrid_data = parse_textgrid_file(file_path)
    
    if 'words' not in textgrid_data:
        raise ValueError("TextGrid文件中未找到'words' tier")
    
    return textgrid_data['words']


def get_non_empty_intervals(intervals: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    过滤掉文本为空的intervals
    
    Args:
        intervals: interval列表
        
    Returns:
        过滤后的interval列表
    """
    return [interval for interval in intervals if interval['text'].strip() != '']


def get_phone_sequence(file_path: str, include_silence: bool = False) -> List[str]:
    """
    获取phone序列
    
    Args:
        file_path: TextGrid文件路径
        include_silence: 是否包含静音标记(sil, sp等)
        
    Returns:
        phone序列列表
    """
    phone_intervals = get_phone_intervals(file_path)
    
    if not include_silence:
        # 过滤掉静音和空格
        phone_intervals = [interval for interval in phone_intervals 
                          if interval['text'].strip() not in ['', 'sil', 'sp']]
    
    return [interval['text'] for interval in phone_intervals]


def get_word_sequence(file_path: str) -> List[str]:
    """
    获取word序列
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        word序列列表
    """
    word_intervals = get_word_intervals(file_path)
    word_intervals = get_non_empty_intervals(word_intervals)
    
    return [interval['text'] for interval in word_intervals]


def load_phone_list(phone_list_path: str) -> List[str]:
    with open(phone_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# 示例用法
if __name__ == "__main__":
    # 假设有一个TextGrid文件
    # file_path = "example.TextGrid"
    source_annot_dir = sys.argv[1]
    phone_list_path = sys.argv[2]
    target_json_path = sys.argv[3]
    
    # 解析整个文件
    # textgrid_data = parse_textgrid_file(file_path)
    # print("所有tiers:", list(textgrid_data.keys()))
    
    # 获取phone intervals
    # phone_intervals = get_phone_intervals(file_path)
    # print(f"Phone intervals数量: {len(phone_intervals)}")
    
    # 获取phone序列
    # phone_sequence = get_phone_sequence(file_path)
    # print("Phone序列:", phone_sequence)
    
    # 获取word序列
    # word_sequence = get_word_sequence(file_path)
    # print("Word序列:", word_sequence)
    phone_list = load_phone_list(phone_list_path)
    print(f"phone_list: {phone_list}")
    make_human_label_json(source_annot_dir, target_json_path, phone_list)
    
    pass

