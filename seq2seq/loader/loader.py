import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np 

class CustomDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	def __init__(self, wordDict, datafile): #, wordDictFile): #, labeled=True, needLabel=True):
		super(CustomDataset, self).__init__()
		print('- dataset: '+datafile)

		self.data = self.readData(datafile)

		with open(wordDict,"rb") as fp:
			self.wordDict = pickle.load(fp,encoding='latin1')
		self.sos_id = 2 
		self.eos_id = 3
		self.unk_id = 1

	def readData(self,datafile):
		question = []
		response = []

		with open(datafile, 'r') as f:
			lines = f.readlines()

		for i in range(len(lines)):
			sentences = lines[i].lower().split('__eou__')[:-1] # there's one empty sentence in the end
		
			for j in range(len(sentences)-1):
				question.append(sentences[j].strip().split())
				response.append(sentences[j+1].strip().split())

		return question, response

	def __len__(self):
		return len(self.data[0])

	def __getitem__(self, idx):
		question_idx = self.word2index(self.data[0][idx])
		response_idx = self.word2index(self.data[1][idx])

		return (question_idx,response_idx)

	def word2index(self, sentence):
		indArr = []
		indArr.append(self.sos_id)
		for i in range(len(sentence)):
			word = sentence[i]
			if word in self.wordDict:
				indArr.append(self.wordDict[word])
			else:
				indArr.append(self.unk_id)
		indArr.append(self.eos_id) 
		indArr = np.array(indArr)
		return indArr
		
def seq_collate(batch):
	batchSize = len(batch)

	maxLen_q = 0
	maxLen_r = 0
	lengths = []

	for i, seq in enumerate(batch):
		seqLen_q = len(seq[0])
		seqLen_r = len(seq[1])
		lengths.append([i, seqLen_q, seqLen_r])
		if seqLen_q > maxLen_q:
			maxLen_q = seqLen_q
		if seqLen_r > maxLen_r:
			maxLen_r = seqLen_r
	question = np.zeros([batchSize, maxLen_q])
	response = np.zeros([batchSize, maxLen_q])
	qLengths = []
	rLengths = []

	lengths = sorted(lengths, key=lambda x:x[1], reverse=True)
	for i in range(batchSize):
		question[i][:lengths[i][1]] = batch[lengths[i][0]][0]
		response[i][:lengths[i][2]] = batch[lengths[i][0]][1]
		rLengths.append(lengths[i][2])
	qLengths = [lengths[i][1] for i in range(len(lengths))]

	question = torch.LongTensor(question)
	response = torch.LongTensor(response)
	qLengths = torch.tensor(qLengths)
	rLengths = torch.tensor(rLengths)


	return {'question': question,
			'qLengths': qLengths,
			'response': response,
			'rLengths': rLengths
		}

class LoaderHandler(object):
	"""docstring for LoaderHandler"""
	def __init__(self, wordDict, data_paths, batch_size):
		super(LoaderHandler, self).__init__()
		print('loader handler...')	

		testData = CustomDataset(wordDict, data_paths['test'])
		self.ldTestEval = DataLoader(testData,batch_size=1, shuffle=False, collate_fn=seq_collate)

		trainData = CustomDataset(wordDict, data_paths['train'])
		self.ldTrain = DataLoader(trainData,batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=seq_collate)

		devData = CustomDataset(wordDict, data_paths['dev'])
		self.ldDev = DataLoader(devData,batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)
