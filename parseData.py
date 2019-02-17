#coding=utf-8
import json

validation_data = "dialogues_validation.txt"
validation_labels = "dialogues_emotion_validation.txt"
with open(validation_data, 'r') as f:
    data_lines = f.readlines()

with open(validation_labels, 'r') as f:
    labels_lines = f.readlines()

result = []

for i in range(len(data_lines)):
    sentences = data_lines[i].strip().split('__eou__')[:-1] #there's an empty sentence in the end
    sentences_labeled = []
    for j in range(len(sentences)):
        sentences_labeled.append([sentences[j].strip().lower(),labels_lines[i].strip().split(' ')[j]])
    result.append([sentences_labeled[0],sentences_labeled[1:]])



with open('validation_json', 'w') as f:
    json.dump(result,f)

data = json.load(open('validation_json','r'))
print(data)
print(len(data))