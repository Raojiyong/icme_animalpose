import json_tricks as json
import random

p1_name = "train.json"

with open(p1_name) as anno_file:
    p1_anno = json.load(anno_file)

n = 5
train_ratio = 0.95
for i in range(n):
    random.shuffle(p1_anno)
    random.shuffle(p1_anno)
    json.dump(p1_anno[:int(len(p1_anno) * train_ratio)], open("train" + str(i + 1) + ".json", 'w'), indent=4)
    json.dump(p1_anno[int(len(p1_anno) * train_ratio):], open("test" + str(i + 1) + ".json", 'w'), indent=4)
