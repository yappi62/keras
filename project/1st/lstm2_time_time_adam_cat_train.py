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
EPOCHS = 20
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
	return rX, rY

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

Y = []
bY = []
for i in range(0, len(aY)):
	code = 0
	YY = np.zeros((len(aY[i]), vocab_size), dtype=np.bool)
	for j in range(0, len(aY[i])):
		YY[j, aY[i][j]] = 1
		code += (aY[i][j] * (vocab_size ** j))
	Y.append(YY)
	bY.append(code)

data = Counter(bY)
accmode = data.most_common(1).pop(0)[1]/float(len(aX))
tY = []
tbY = []
for i in range(0, len(taY)):
	code = 0
	tYY = np.zeros((len(taY[i]), vocab_size), dtype=np.bool)
	for j in range(0, len(taY[i])):
		tYY[j, taY[i][j]] = 1
		code += (taY[i][j] * (vocab_size ** j))
	tY.append(tYY)
	tbY.append(code)

data = Counter(tbY)
taccmode = data.most_common(1).pop(0)[1]/float(len(taX))

print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))

print('Build model ...')
model = Sequential()
model.add(Embedding(vocab_size, EMBED_SIZE, mask_zero=True))
model.add(LSTM(EMBED_SIZE, HIDDEN_SIZE))
model.add(RepeatVector(ans_maxlen))
model.add(LSTM(HIDDEN_SIZE, HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributedDense(HIDDEN_SIZE, vocab_size)) # TimeDistributedDense
model.add(Activation('time_distributed_softmax')) # time_distributed_softmax

model.compile(optimizer='adam', loss='categorical_crossentropy')

print('Training ...')
begin = datetime.now()
for i in range(0, len(aX)):
	model.train_on_batch(np.array([aX[i]]), [Y[i]], accuracy=True)

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

fResult = open('result5.txt', 'a+')
fResult.write('Question type = %s\n'%(quesTypes))
fResult.write('Average learning time = %ddays %d:%d:%.2f\n'%(avgDay, avgHour, avgMin, avgSec))
fResult.write('Test loss / test accuracy = {:.4f} / {:.4f}\n'.format(loss, acc))
fResult.write('mode acc / test mode acc = {:.4f} / {:.4f}\n\n'.format(accmode, taccmode))
fResult.close()

