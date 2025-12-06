import json
import os
import re

# patter = re.compile('^.*?LibriSpeech/')
# new_file = open('datalist.txt', 'w')
# with open('train.100.phone.datalist') as tf:
    # for line in tf.readlines():
        # obj = json.loads(line.strip())
        # wav_obj = obj['sph'] 
        # wav_obj = 'LibriSpeech/'+patter.sub('', wav_obj)
        # obj.update({'sph': wav_obj})
        # if os.path.isfile(wav_obj):
            # new_file.write(f"{json.dumps(obj)}\n")


"""
  data_list: datalist.txt
  valid_list: dev.360.phone.datalist

"""

new_list = open('datalist.txt.phn', 'w')
with open('datalist.txt') as df:
    for line in df.readlines():
        obj = json.loads(line.strip())
        label = obj['label']
        kw_candidate = [x for x in range(len(label))]
        obj.update({'phn_label': label, 'kw_candidate': kw_candidate})
        new_list.write(f"{json.dumps(obj)}\n")
