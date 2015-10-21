from __future__ import absolute_import
from __future__ import print_function
from vqaTools.vqa import VQA
import random

from functools import reduce
import re
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, RepeatVector
from keras.layers.recurrent import LSTM, JZS1
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.utils import np_utils, generic_utils
from keras.callbacks import Callback
import numpy as np
np.random.seed(1337)  # for reproducibility
from datetime import datetime
from collections import Counter


# EACH_LIMIT_SIZE = 10000
# VAL_EACH_LIMIT_SIZE = 1000
LIMIT_SIZE = 10000
VAL_LIMIT_SIZE = 1000
EMBED_SIZE = 500
HIDDEN_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 300
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

def vectorize(data, size):
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


print('Enter the quesTypes (\'what color\', \'is this\', ..., \'all\')')
# quesTypes = input()
quesTypes = 'is this'

if quesTypes == 'all':
	annIdsA = vqa.getQuesIds()
	tannIdsA = tvqa.getQuesIds()
else:
	annIdsA = vqa.getQuesIds(quesTypes = quesTypes)
	tannIdsA = tvqa.getQuesIds(quesTypes = quesTypes)

'''
annIdsA = annIds[0:EACH_LIMIT_SIZE]
tannIdsA = tannIds[0:VAL_EACH_LIMIT_SIZE]

quesTypes = 'where'
annIds = vqa.getQuesIds(quesTypes = quesTypes)
tannIds = tvqa.getQuesIds(quesTypes = quesTypes)
annIdsA += annIds[0:EACH_LIMIT_SIZE]
tannIdsA += tannIds[0:VAL_EACH_LIMIT_SIZE]

quesTypes = 'what is'
annIds = vqa.getQuesIds(quesTypes = quesTypes)
tannIds = tvqa.getQuesIds(quesTypes = quesTypes)
annIdsA += annIds[0:EACH_LIMIT_SIZE]
tannIdsA += tannIds[0:VAL_EACH_LIMIT_SIZE]
'''

annsA = vqa.loadQA(annIdsA)
tannsA = tvqa.loadQA(tannIdsA)
	
train = get_inputList(vqa, annsA, LIMIT_SIZE)
test = get_inputList(tvqa, tannsA, VAL_LIMIT_SIZE)
vocab = sorted(list(set(train + test)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 2) for i, c in enumerate(vocab))

train = get_inputVec(vqa, annsA, LIMIT_SIZE)
test = get_inputVec(tvqa, tannsA, VAL_LIMIT_SIZE)

ques_maxlen = max(map(len, (x for x, _ in train + test)))
ans_maxlen = max(map(len, (x for _, x in train + test))) + 1 # +1 because of adding END_MARK

# ques_maxlen = 100
# ans_maxlen = 20
# vocab_size = 10000

aX, aY = vectorize(train, LIMIT_SIZE)
taX, taY = vectorize(test, VAL_LIMIT_SIZE)

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
			tX[i, j, taX[i, j]-1] = 1


for i in range(0, len(taY)):
	start = False
	code = 0
	count = 0
	for j in range(0, ans_maxlen):
		if start == False:
			if taY[i, j] != 0:
				start = True
		if start == True:
			tY[i, j, taY[i, j]-1] = 1
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
model.add(Embedding(vocab_size+1, EMBED_SIZE, mask_zero=True))
model.add(JZS1(EMBED_SIZE, HIDDEN_SIZE))
model.add(Dense(HIDDEN_SIZE, HIDDEN_SIZE))
model.add(RepeatVector(ans_maxlen))
model.add(JZS1(HIDDEN_SIZE, HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributedDense(HIDDEN_SIZE, vocab_size, activation="time_distributed_softmax")) # TimeDistributedDense
# model.add(Activation('softmax')) # time_distributed_softmax

model.compile(optimizer='adam', loss='categorical_crossentropy') # mean_squared_error, categorical_crossentropy

print('Training ...')
fResult = open('jzs1_embed_den_time_time_pre_adam_cat.txt', 'w')
fResult.write('Question type = %s\n'%(quesTypes))
fResult.write('BatchSize %d\n'%(BATCH_SIZE))
fResult.write('Epochs %d\n'%(EPOCHS))
fResult.write('VocabSize %d\n'%(vocab_size))
fResult.close()
class LossHistory(Callback):
	def on_epoch_end(self, epoch, logs={}):
		fResult = open('jzs1_embed_den_time_time_pre_adam_cat.txt', 'a+')
		loss = logs.get('val_loss')
		acc = logs.get('val_acc')
		fResult.write('val %d %.4f %.4f\n'%(epoch, loss, acc))
		fResult.close()
	def on_batch_end(self, batch, logs={}):
		fResult = open('jzs1_embed_den_time_time_pre_adam_cat.txt', 'a+')
		loss = logs.get('loss')
		acc = logs.get('acc')
		fResult.write('train %d %.4f %.4f\n'%(batch, loss, acc))
		fResult.close()


begin = datetime.now()

history = LossHistory()
model.fit(aX, Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data=(taX, tY), verbose=1, show_accuracy=True, callbacks=[history])

end = datetime.now()
diff = end - begin
Sec = diff.total_seconds()
Min = int(Sec/60)
Hour = int(Min/60)
Day = int(Hour/24)
Sec -= 60*Min
Min -= 60*Hour
Hour -= 24*Day

model.save_weights('jzs1_embed_den_time_time_pre_adam_cat.hdf5', overwrite=True)

print('X.shape = {}'.format(X.shape))
print('Y.shape = {}'.format(Y.shape))
print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))
print('mode acc / test mode acc = %.4f / %.4f\n'%(accmode, taccmode))
print('Total learning time = %ddays %d:%d:%d\n\n'%(Day, Hour, Min, Sec))

fResult = open('jzs1_embed_den_time_time_pre_adam_cat.txt', 'a+')
fResult.write('mode acc / test mode acc = %.4f / %.4f\n'%(accmode, taccmode))
fResult.write('Total learning time = %ddays %d:%d:%d\n\n'%(Day, Hour, Min, Sec))
fResult.close()
