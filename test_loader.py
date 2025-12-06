import os
import json
import sys
import yaml
import torchaudio
import torch
import random

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from yamlinclude import YamlIncludeConstructor
from data.loader.data_loader import Dataset
from local.utils import  read_list, make_dict_from_file
from model.TransformerKWSPhone import TransformerKWSPhone



def run(config):
    data_config = config['data_config']
    data_list_file = data_config['data_list']
    model_config = config['model_config']
    model = TransformerKWSPhone(**model_config)
    tr_list = read_list(data_list_file, split_cv=False, shuffle=True)
    num_worker = data_config.get('num_worker', 10)

    if num_worker > 10:
        num_worker = 6

    tr_set = Dataset(
        data_config,
        tr_list,
    )
    tr_loader = DataLoader(
        tr_set,
        batch_size=None,
        num_workers=num_worker
    )
    with torch.no_grad():
        for e in range(5):
            for batch_id, data in enumerate(tr_loader):
                mixspeech,mixspeech_len,phn_label,phn_label_len,keyword,keyword_len,md_label,md_label_len,target = data
                print (target, keyword, md_label, phn_label)
                print ("======="*20)
                #print (phn_label.view(-1), phn_label_len)
                # model((mixspeech,mixspeech_len,phn_label,phn_label_len,keyword,keyword_len,md_label,md_label_len,target))
                # print ("===========")
                # print (mixspeech.size())
                # for i, x in enumerate(mixspeech):
                    # x = x.view(1,-1)
                    # torchaudio.save(f'{i}.wav', x, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
                # exit()
                # print ("\n\n\n")
                # print (mixspeech.size(), phn_label.size())
                # print (phn_label)
        
def make_mix_permuate(batch_size):
    n_mix = [1,2,3]
    n_mix = random.choices(n_mix, weights=n_mix, k=1)[0]
    raw_idx = [x for x in range(batch_size)]
    idxs = [raw_idx]
    for i in range(1, n_mix):
        head_idx = raw_idx[0:i]
        tail_idx = raw_idx[i:]
        new_idx = tail_idx + head_idx
        idxs.append(new_idx)
    return idxs

def update(org_list):
    new_list = open('debug.new.list', 'w')
    new_root = "/Users/shiying/Research/zeus_dev/conditional_chain_asr/debug_dir/debug_wav/"
    with open(org_list) as lf:
        for line in lf.readlines():
            line = json.loads(line.strip())
            wav_path = line['sph']
            wav_name = os.path.basename(wav_path)
            new_path = f"{new_root}{wav_name}"
            line.update({'sph': new_path})
            new_list.write(f"{json.dumps(line)}\n")
    
if __name__ == '__main__':

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    config = sys.argv[1]
    config = yaml.load(open(config),Loader=yaml.FullLoader)
    run(config)
