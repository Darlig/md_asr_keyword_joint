import shutil
import json

# "/work102/shiying/data/LibriSpeech/train-clean-100/7067/76048/7067-76048-0037.wav"
with open('train.100.phone.datalist') as tf:
    for line in tf.readlines():
        obj = json.loads(line.strip())
        print (obj['sph'])
