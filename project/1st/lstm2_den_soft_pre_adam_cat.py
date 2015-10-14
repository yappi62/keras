from __future__ import absolute_import
from __future__ import print_function
from vqaTools.vqa import VQA
import random

from functools import reduce
import re
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.utils import np_utils, generic_utils
import numpy as np
from datetime import datetime
from collections import Counter


LIMIT_SIZE = 10000
EMBED_SIZE = 300
HIDDEN_SIZE = 512
BATCH_SIZE = 128
EPOCHS = 500
END_MARK = 1

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
	return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

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
		rY.append(y+[END_MARK])
		limit += 1
		if limit == LIMIT_SIZE :
			break
	return pad_sequences(rX, maxlen=ques_maxlen), pad_sequences(rY, maxlen=ans_maxlen)


print('Enter the quesTypes (\'what color\', \'is this\', ..., \'all\')')
# quesTypes = input()
quesTypes = 'is this'

if quesTypes == 'all':
	annIdsA = vqa.getQuesIds()
	tannIdsA = tvqa.getQuesIds()
else:
	annIdsA = vqa.getQuesIds(quesTypes = quesTypes)
	tannIdsA = tvqa.getQuesIds(quesTypes = quesTypes)


annsA = vqa.loadQA(annIdsA)
tannsA = tvqa.loadQA(tannIdsA)
	
train = get_inputList(vqa, annsA)
test = get_inputList(tvqa, tannsA)
vocab = sorted(list(set(train + test)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

train = get_inputVec(vqa, annsA)
test = get_inputVec(tvqa, tannsA)

ques_maxlen = max(map(len, (x for x, _ in train + test)))
ans_maxlen = max(map(len, (x for _, x in train + test)))

# ques_maxlen = 100
# ans_maxlen = 20
# vocab_size = 10000

aX, aY = vectorize(train)
taX, taY = vectorize(test)

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
			X[i, j, aX[i, j]] = 1

for i in range(0, len(aY)):
	start = False
	code = 0
	count = 0
	for j in range(0, ans_maxlen):
		if start == False:
			if aY[i, j] != 0:
				start = True
		if start == True:
			Y[i, j, aY[i, j]] = 1
			code += (aY[i, j] * (vocab_size ** (count)))
			count += 1
	bY.append(code)

data = Counter(bY)
accmode = data.most_common(1).pop(0)[1]/float(len(aX))
tX = np.zeros((len(taX), ques_maxlen, vocab_size), dtype=np.bool)
tY = np.zeros((len(taY), ans_maxlen, vocab_size), dtype=np.bool)
tbY = []
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
	code = 0
	count = 0
	for j in range(0, ans_maxlen):
		if start == False:
			if taY[i, j] != 0:
				start = True
		if start == True:
			tY[i, j, taY[i, j]] = 1
			code += (taY[i, j] * (vocab_size ** (count)))
			count += 1
	tbY.append(code)

data = Counter(tbY)
taccmode = data.most_common(1).pop(0)[1]/float(len(taX))

print('X.shape = {}'.format(X.shape))
print('Y.shape = {}'.format(Y.shape))
print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))

print('Build model ...')
model = Sequential()
model.add(Embedding(vocab_size, EMBED_SIZE, mask_zero=True))
model.add(LSTM(EMBED_SIZE, HIDDEN_SIZE))
model.add(RepeatVector(ans_maxlen))
model.add(LSTM(HIDDEN_SIZE, HIDDEN_SIZE, return_sequences=True))
model.add(Dense(HIDDEN_SIZE, vocab_size)) # TimeDistributedDense
model.add(Activation('softmax')) # time_distributed_softmax

model.compile(optimizer='adam', loss='categorical_crossentropy')

print('Training ...')
fResult = open('lstm2_den_soft_pre_adam_cat.txt', 'a+')
fResult.write('Question type = %s\n'%(quesTypes))
fResult.close()
# model.fit(aX, Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.1, show_accuracy=True)

begin = datetime.now()
for i in range(0, EPOCHS):
	index = 0
	print('-'*40)
	print('EPOCH', i)
	print('-'*40)
	progbar = generic_utils.Progbar(aX.shape[0])
	for j in range(len(aX)/BATCH_SIZE):
		fResult = open('lstm2_den_soft_pre_adam_cat.txt', 'a+')
		loss, acc = model.train_on_batch(aX[index:index+BATCH_SIZE], Y[index:index+BATCH_SIZE], accuracy=True)
		progbar.add(BATCH_SIZE, values=[("train loss", loss), ("train acc", acc)])
		fResult.write('train %d %d %.4f %.4f\n'%(i, j, loss, acc))
		index += BATCH_SIZE
		fResult.close()

end = datetime.now()
diff = end - begin
avgSec = diff.total_seconds()/EPOCHS
avgMin = int(avgSec/60)
avgHour = int(avgMin/60)
avgDay = int(avgHour/24)
avgSec -= 60*avgMin
avgMin -= 60*avgHour
avgHour -= 24*avgDay

# loss, acc = model.evaluate(taX, tY, batch_size=BATCH_SIZE, show_accuracy=True)
progbar = generic_utils.Progbar(taX.shape[0])
index = 0
for i in range(len(taX)/BATCH_SIZE):
	loss, acc = model.test_on_batch(taX[index:index+BATCH_SIZE], tY[index:index+BATCH_SIZE], accuracy=True)
	progbar.add(BATCH_SIZE, values=[("test loss", loss), ("test acc", acc)])
	index += BATCH_SIZE

fResult = open('lstm2_den_soft_pre_adam_cat.txt', 'a+')
fResult.write('Test loss / test accuracy = %.4f / %.4f\n'%(loss, acc))
fResult.write('mode acc / test mode acc = %.4f / %.4f\n'%(accmode, taccmode))
fResult.write('Average learning time = %ddays %d:%d:%d\n\n'%(avgDay, avgHour, avgMin, avgSec))
fResult.close()
