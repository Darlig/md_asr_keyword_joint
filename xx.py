import torch
import torch.nn as nn
from typing_extensions import List
import json
from data.loader.data_utils import unfold_list

utt2wav = {}
with open('l2.scp') as lf:
    for line in lf.readlines():
        line = line.strip()
        utt, wav = line.split(" ")
        wav = wav.replace(".wav", '.16.wav')
        utt2wav.update({utt:wav})

x = open('l2.test.datalist', 'w')
with open('l2.text.phn') as lf:
    for line in lf.readlines():
        line = line.strip()
        obj = json.loads(line)
        obj.update({'sph': utt2wav[obj['key']]}) 
        x.write(f"{json.dumps(obj)}\n")
# sym2phone = {}
# with open('sym2id.txt') as sf:
#     for line in sf.readlines():
#         line = line.strip()
#         obj = json.loads(line)
#         sym = obj['sym']
#         phn_id = unfold_list(obj['labels'])
#         for idx, phn in enumerate(phn_id):
#             if sym[idx] not in sym2phone:
#                 sym2phone[sym[idx]] = phn
#             else:
#                 continue
#         #if len(sym) != len(phn_id):
#         #    print ("xxx")
#         ##print (len(sym), len(phn_id))
# word2phone = {}
# with open('librispeech-lexicon.txt') as lf:
#     for line in lf.readlines():
#         word, phone_seq = line.strip().split(" ", maxsplit=1)
#         if word not in word2phone:
#             word2phone[word] = phone_seq
#         #else:
#         #    word2phone[word].append(phone_seq)



# #for word, phone_seq in word2phone.items():
# #    print (word, phone_seq)

# #with open('human_scores.json') as hf:
# #    for line in hf.readlines():
# #        line = line.strip()
# timit_test = open('timit.text.phn', 'w')
# obj = json.load(open('human_scores.json'))
# for key, value in obj.items():
#     phones =  value['words'][0]['phones']
#     md_label =  value['words'][0]['phones-accuracy']
#     word_seq = value['words'][0]['text'].upper()
#     phn_seq = []
#     for word in word_seq.split(" "):
#         xx=word2phone[word]
#         phn_seq.extend(xx.split(" "))
#     if len(phn_seq) - len(phones) != 0:
#         continue
    
#     timit_test.write(f"{json.dumps({'key': key, 'phn_label':[sym2phone[p] for p in phn_seq],'md_label': md_label})}\n")
    #if len(phn_seq) != len(phones):
    #    print (phn_seq, "xxxxx", phones)
    #print (value['words'][0]['text'])
    #for phone in phones:
    #    if phone not in sym2phone:
    #        print (phone)
    #print (value['words']['phones'])
#for i, one_obj in enumerate(obj):
#    print (one_obj['words']['phones'])