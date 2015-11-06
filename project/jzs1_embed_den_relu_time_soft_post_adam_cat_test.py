from __future__ import absolute_import
from __future__ import print_function

from functools import reduce
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, RepeatVector, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, JZS1
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.callbacks import Callback

import numpy as np
np.random.seed(1337)  # for reproducibility
from datetime import datetime
from collections import Counter
import random
import re
from preprocess_util import *

# import VQA api
from vqaTools.vqa import VQA

# import VGG 16-layer network structure
from vgg_16_keras import VGG_16


##### Initialize parameters

LIMIT_SIZE = 500
VAL_LIMIT_SIZE = 100
EMBED_SIZE = 500
HIDDEN_SIZE = 500
BATCH_SIZE = 4
EPOCHS = 2

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


##### initialize VQA api for QA annotations

vqa=VQA(annFile, quesFile)	# training
tvqa=VQA(tannFile, tquesFile)	# validation

# QA annotations for given question types
"""
quesTypes can be one of the following
..
what color 	what kind 	what are 	what type  	is the
is this		how many 	are 		does  		where
is there 	why 		which		do 		what does 
what time 	who 		what sport 	what animal 	what brand
"""

##### Load VQA dataset

print('Enter the quesTypes (\'what color\', \'is this\', ..., \'all\')')
# quesTypes = input()
quesTypes = 'what color'

if quesTypes == 'all':
	annIdsA = vqa.getQuesIds()
	tannIdsA = tvqa.getQuesIds()
	#imgIdsA = vqa.getImgIds()
	#timgIdsA = tvqa.getImgIds()
else:
	annIdsA = vqa.getQuesIds(quesTypes = quesTypes)
	tannIdsA = tvqa.getQuesIds(quesTypes = quesTypes)
	#imgIdsA = vqa.getImgIds(quesTypes = quesTypes)
	#timgIdsA = tvqa.getImgIds(quesTypes = quesTypes)

annsA = vqa.loadQA(annIdsA)
tannsA = tvqa.loadQA(tannIdsA)

if len(annsA) > LIMIT_SIZE:
	annsA[LIMIT_SIZE:] = []
	#imgIdsA[LIMIT_SIZE:] = []

if len(tannsA) > VAL_LIMIT_SIZE:
	tannsA[VAL_LIMIT_SIZE:] = []
	#timgIdsA[VAL_LIMIT_SIZE:] = []

train = get_inputList(vqa, annsA)
test = get_inputList(tvqa, tannsA)
vocab = sorted(list(set(train + test)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 2) for i, c in enumerate(vocab))

train = get_inputVec(vqa, annsA)
test = get_inputVec(tvqa, tannsA)

ques_maxlen = max(map(len, (x for x, _ in train + test)))
ans_maxlen = max(map(len, (x for _, x in train + test))) + 1 # +1 because of adding END_MARK



##### Build Train/Test matrices for text QA

print('Building Train/Val matrices')
aX, aY = vectorize(train, word_idx, ques_maxlen, ans_maxlen)
taX, taY = vectorize(test, word_idx, ques_maxlen, ans_maxlen)

X_text, Y, bY, wY= buildMat_text(aX, aY, ques_maxlen, ans_maxlen, vocab_size)
data = Counter(bY)
accmode = data.most_common(1).pop(0)[1]/float(len(aX))
# X_img = buildMat_img(imgIdsA, imgDir, 'train2014')
# X = [X_img, aX]		# Use aX instead of X_text in training

tX_text, tY, tbY, twY = buildMat_text(taX, taY, ques_maxlen, ans_maxlen, vocab_size)
data = Counter(tbY)
taccmode = data.most_common(1).pop(0)[1]/float(len(taX))
# tX_img = buildMat_img(timgIdsA, timgDir, 'val2014')
# tX = [tX_img, taX]

print('X_text.shape = {}'.format(X_text.shape))
# print('X_img.shape = {}'.format(X_img.shape))
print('Y.shape = {}'.format(Y.shape))
print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))



print('Build model ...')

'''
### Load VGGNet (CNN)
vggnet = VGG_16('vgg16_weights.h5')	# download(553MB) site: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 --> weights
vggnet.layers.pop()	# pop last Dense layer to connect to fc7
vggnet.layers.pop()	# pop Dropout layer
vggnet.params.pop()	
vggnet.params.pop()
vggnet.add(Dense(4096, HIDDEN_SIZE, activation="relu")) # to match the dimensions of image-question features (in Merge layer)
'''

### Build Question RNN
model = Sequential()
model.add(Embedding(vocab_size+1, EMBED_SIZE, mask_zero=True))
model.add(JZS1(EMBED_SIZE, HIDDEN_SIZE))

### Merged model
# model = Sequential()
# model.add(Merge([vggnet, qnet], mode="concat", concat_axis=1)) # output_dim = 2*HIDDEN_SIZE
model.add(Dense(HIDDEN_SIZE, HIDDEN_SIZE, activation="relu"))
model.add(RepeatVector(ans_maxlen))
model.add(JZS1(HIDDEN_SIZE, HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributedDense(HIDDEN_SIZE, vocab_size, activation="softmax")) # TimeDistributedDense

print('Model compiling ...')
#opt = Adam(lr = 0.000125)
model.compile(optimizer='adam', loss='categorical_crossentropy') # mean_squared_error, categorical_crossentropy




##### Training (+logging)

print('Training ...')

begin = datetime.now()

history = LossHistory()
model.fit(aX, Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data=(taX, tY, twY), verbose=1, show_accuracy=True, sample_weight=wY)
pY = model.predict(taX, batch_size=BATCH_SIZE)
ppY = np.zeros((len(pY), ans_maxlen, vocab_size), dtype=np.bool)

fPredict = open('test_pred.txt', 'w')
for i in range(0, len(pY)):
	fPredict.write('\n')
	for j in range(0, ans_maxlen):
		index = aY[i, j]
		if index == 1:
			fPredict.write(u'  / ')
			break
		else:
			for word, idx in word_idx.iteritems():
				if idx == index:
					fPredict.write(u' '+word)
					break
	for j in range(0, ans_maxlen):
		pmax = pY[i, j, 0]
		index = 0
		for k in range(1, vocab_size):
			if(pY[i, j, k] > pmax):
				pmax = pY[i, j, k]
				index = k
		ppY[i, j, index] = True
		if index == 0:
			fPredict.write(u' end'+'(%.4f)'%(pmax))
		for word, idx in word_idx.iteritems():
			if idx == (index+1):
				fPredict.write(u' '+word+'(%.4f)'%(pmax))
				break
fPredict.close()

nacc = 0
nMask = 0
for i in range(0, len(pY)):
	for j in range(0, ans_maxlen):
		if(wY[i, j, 0] == True & wY[i, min(j+1, ans_maxlen-1), 0]) == True:
			nMask += 1
			dot = np.dot(ppY[i, j], tY[i, j])
			if dot == True:
				nacc += 1

acc = float(nacc)/nMask

end = datetime.now()
diff = end - begin
Sec = diff.total_seconds()
Min = int(Sec/60)
Hour = int(Min/60)
Day = int(Hour/24)
Sec -= 60*Min
Min -= 60*Hour
Hour -= 24*Day


#print('X.shape = {}'.format(X.shape))
#print('Y.shape = {}'.format(Y.shape))
#print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))
print('mode acc / test mode acc = %.4f / %.4f\n'%(accmode, taccmode))
print('Total learning time = %ddays %d:%d:%d\n\n'%(Day, Hour, Min, Sec))