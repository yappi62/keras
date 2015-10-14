from __future__ import absolute_import
from __future__ import print_function
from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

from functools import reduce
import re
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import numpy as np
from datetime import datetime


MAX_SIZE = 5000
HIDDEN_SIZE = 512
BATCH_SIZE = 128
EPOCHS = 20

dataDir='../../VQA'
# print('Enter the taskType (\'OpenEnded\', \'MultipleChoice\')')
# taskType=input()
taskType = 'OpenEnded'
dataType='mscoco'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, 'train2014')
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, 'train2014')
imgDir = '%s/Images/%s/' %(dataDir, 'train2014')
tannFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, 'val2014')
tquesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, 'val2014')
timgDir = '%s/Images/%s/' %(dataDir, 'val2014')

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)
tvqa=VQA(tannFile, tquesFile)

# load and display QA annotations for given question types
"""
quesTypes can be one of the following
what color 
what kind 
what are 
what type 
is the 
is this
how many 
are 
does 
where 
is there 
why 
which 
do 
what does 
what time 
who 
what sport 
what animal 
what brand
"""

def tokenize(sent):
	'''Return the tokens of a sentence including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	'''
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def get_inputList(vqa, anns):
	data = []
	for i in range(0, len(anns)):
		ques = tokenize(vqa.qqa[anns[i]['question_id']]['question'])
		ans = tokenize(anns[i]['multiple_choice_answer'])
		data += (ques + ans)
	return data

def get_inputVec(vqa, anns):
	data = []
	for i in range(0, len(anns)):
		ques = tokenize(vqa.qqa[anns[i]['question_id']]['question'])
		ans = tokenize(anns[i]['multiple_choice_answer'])
		data.append((ques, ans))
	return data

def vectorize(data):
	rX = []
	rY = []
	limit = 0
	for ques, ans in data:
		x = [word_idx[w] for w in ques]
		y = [word_idx[w] for w in ans]
		rX.append(x)
		rY.append(y)
		limit += 1
		if limit == MAX_SIZE :
			break
	
	return pad_sequences(rX, maxlen=ques_maxlen), pad_sequences(rY, maxlen=ques_maxlen)

annIdsA = vqa.getQuesIds()
annsA = vqa.loadQA(annIdsA)

tannIdsA = tvqa.getQuesIds()
tannsA = tvqa.loadQA(tannIdsA)

train = get_inputVec(vqa, annsA)
test = get_inputVec(tvqa, tannsA)

ques_maxlen = max(map(len, (x for x, _ in train + test)))
ans_maxlen = max(map(len, (x for _, x in train + test)))

print('Enter the quesTypes (\'what color\', \'is this\', ..., \'all\')')
# quesTypes = input()
quesTypes = 'what color'

if quesTypes != 'all':
	annIdsA = vqa.getQuesIds(quesTypes = quesTypes)
	annsA = vqa.loadQA(annIdsA)
	tannIdsA = tvqa.getQuesIds(quesTypes = quesTypes)
	tannsA = tvqa.loadQA(tannIdsA)
	
	train = get_inputList(vqa, annsA)
	test = get_inputList(tvqa, tannsA)
	vocab = sorted(list(set(train + test)))
	# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
	
	train = get_inputVec(vqa, annsA)
	test = get_inputVec(tvqa, tannsA)


aX, aY = vectorize(train)
taX, taY = vectorize(test)

X = np.zeros((len(aX), ques_maxlen, vocab_size), dtype=np.bool)
Y = np.zeros((len(aY), ques_maxlen, vocab_size), dtype=np.bool)
for i in range(0, len(aX)):
	start = False
	for j in range(0, ques_maxlen):
		if start == False:
			if aX[i, j] != 0:
				start = True
		if start == True:
			X[i, j, aX[i, j]] = 1

for i in range(0, len(aY)):
	start = False
	for j in range(0, ques_maxlen):
		if start == False:
			if aY[i, j] != 0:
				start = True
		if start == True:
			Y[i, ques_maxlen-1-j, aY[i, j]] = 1

tX = np.zeros((len(taX), ques_maxlen, vocab_size), dtype=np.bool)
tY = np.zeros((len(taY), ques_maxlen, vocab_size), dtype=np.bool)
for i in range(0, len(taX)):
	start = False
	for j in range(0, ques_maxlen):
		if start == False:
			if taX[i, j] != 0:
				start = True
		if start == True:
			tX[i, j, taX[i, j]] = 1

for i in range(0, len(taY)):
	start = False
	for j in range(0, ques_maxlen):
		if start == False:
			if taY[i, j] != 0:
				start = True
		if start == True:
			tY[i, ques_maxlen-1-j, taY[i, j]] = 1
	

print('X.shape = {}'.format(X.shape))
print('Y.shape = {}'.format(Y.shape))
print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))

print('Build model ...')
model = Sequential()
model.add(LSTM(vocab_size, HIDDEN_SIZE, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(HIDDEN_SIZE, vocab_size))
model.add(Activation('time_distributed_softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')

print('Training ...')
begin = datetime.now()
model.fit(X, Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05, show_accuracy=True)
end = datetime.now()
diff = end - begin
avgSec = diff.total_seconds()/EPOCHS
avgMin = int(avgSec/60)
avgHour = int(avgMin/60)
avgDay = int(avgHour/24)
avgSec -= 60*avgMin
avgMin -= 60*avgHour
avgHour -= 24*avgDay

loss, acc = model.evaluate(tX, tY, batch_size=BATCH_SIZE, show_accuracy=True)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

fResult = open('result1.txt', 'a+')
fResult.write('Question type = %s\n'%(quesTypes))
fResult.write('Average learning time = %ddays %d:%d:%.2f\n'%(avgDay, avgHour, avgMin, avgSec))
fResult.write('Test loss / test accuracy = {:.4f} / {:.4f}\n\n'.format(loss, acc))
fResult.close()
