
import os
import csv
import json
import random
random.seed(2023)

DATA_DIR = os.environ['DATA_DIR']
csv_file = os.path.join(DATA_DIR, 'CoT_zh/CoT_Chinese_data.csv')
train_json_file = os.path.join(DATA_DIR, 'CoT_zh/train.json')
test_json_file = os.path.join(DATA_DIR, 'CoT_zh/test.json')

train = {'prompt':[],'query':[],'response':[]}
test = {'prompt':[],'query':[],'response':[]}
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        if random.random() < 0.9: # for train
            train['prompt'].append(line[0])
            train['query'].append(line[1])
            train['response'].append(line[2])
        else:
            test['prompt'].append(line[0])
            test['query'].append(line[1])
            test['response'].append(line[2])

with open(train_json_file,'w',encoding='utf-8') as f:
    json.dump(train,f,ensure_ascii=False,indent=4)
with open(test_json_file,'w',encoding='utf-8') as f:
    json.dump(test,f,ensure_ascii=False,indent=4)

