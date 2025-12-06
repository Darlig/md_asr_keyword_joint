import json
import re

PATTERN = re.compile('^.*?test-clean-ssd-wav/')


test_data_list = open('test.datalist.phn', 'w')

with open('test.clean.phoneid.txt') as tf:
    for line in tf.readlines():
        obj = json.loads(line.strip())
        sph = obj['sph']
        sph = PATTERN.sub("test-clean/", sph)
        phn_label = obj['labels']
        obj.update({'phn_label': phn_label})
        obj.update({'sph': sph})
        test_data_list.write(f"{json.dumps(obj)}\n")
