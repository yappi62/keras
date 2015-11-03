from __future__ import absolute_import
from __future__ import print_function

from functools import reduce
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import re

END_MARK = 1

def tokenize(sent):
	'''Return the tokens of a sentence including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	'''
	return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

def get_inputList(vqa, anns, size):
	data = []
	limit = 0
	for i in range(0, len(anns)):
		ques = tokenize(vqa.qqa[anns[i]['question_id']]['question'])
		ans = tokenize(anns[i]['multiple_choice_answer'])
		data += (ques + ans)
		limit += 1
		if limit == size:
			break
	return data

def get_inputVec(vqa, anns, size):
	data = []
	limit = 0
	for i in range(0, len(anns)):
		ques = tokenize(vqa.qqa[anns[i]['question_id']]['question'])
		ans = tokenize(anns[i]['multiple_choice_answer'])
		data.append((ques, ans))
		limit += 1
		if limit == size:
			break
	return data

def vectorize(data, size, word_idx, ques_maxlen, ans_maxlen):
	rX = []
	rY = []
	limit = 0
	for ques, ans in data:
		x = [word_idx[w] for w in ques]
		y = [word_idx[w] for w in ans]
		rX.append(x)
		rY.append(y+[END_MARK])
		limit += 1
		if limit == size :
			break
	return pad_sequences(rX, maxlen=ques_maxlen), pad_sequences(rY, maxlen=ans_maxlen)

def buildMat(aX, aY, ques_maxlen, ans_maxlen, vocab_size):
	X = np.zeros((len(aX), ques_maxlen, vocab_size), dtype=np.bool)
	Y = np.zeros((len(aY), ans_maxlen, vocab_size), dtype=np.bool)
	bY = []
	for i in range(0, len(aX)):
		start = False
		for j in range(0, ques_maxlen):
			if start == False:
				if aX[i, j] != 0:
					start = True
			if start == True:
				X[i, j, aX[i, j]-1] = 1

	for i in range(0, len(aY)):
		start = False
		code = 0
		count = 0
		for j in range(0, ans_maxlen):
			if start == False:
				if aY[i, j] != 0:
					start = True
			if start == True:
				Y[i, j, aY[i, j]-1] = 1
				code += (aY[i, j] * (vocab_size ** (count)))
				count += 1
		bY.append(code)
	return X, Y, bY
