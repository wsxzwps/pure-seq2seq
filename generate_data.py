train_data = "data/dialogues_train.txt"


with open(train_data, 'r', encoding='utf-8') as f:
    data_lines = f.readlines()

result = []

for i in range(len(data_lines)):
    cleaned_data = bytes(data_lines[i], 'utf-8').decode('utf-8', 'ignore')
    sentences = cleaned_data.strip().split('__eou__')[:-1] #there's an empty sentence in the end
    for j in range(len(sentences) - 1):
        result.append([sentences[j].strip().lower(),sentences[j+1].strip().lower()])




with open('data/train', 'w', encoding='utf-8', errors='ignore') as f:
    for pair in result:
        f.write('\t'.join(pair))
        f.write('\n')

