from __future__ import absolute_import
from __future__ import print_function

from functools import reduce
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, RepeatVector, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, JZS1
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.callbacks import Callback

import os
import numpy as np
import scipy.io as sio
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

LIMIT_ITER = 10
LIMIT_SIZE = 20000
TR_LIMIT_SIZE = LIMIT_ITER*LIMIT_SIZE
VAL_LIMIT_SIZE = 10000
EMBED_SIZE = 500
HIDDEN_SIZE = 500
BATCH_SIZE = 128
lr = 0.1
EPOCHS = 100


dataDir='../../VQA'
imgDir = '%s/Images/%s/' %(dataDir, 'train2014')
timgDir = '%s/Images/%s/' %(dataDir, 'val2014')
vggFeatDir = '%s/coco/' %(dataDir)

taskType = 'OpenEnded'
dataType='mscoco'
quesTypes = ['is this', 'what color', 'what is']

annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, 'train2014')
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, 'train2014')
tannFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, 'val2014')
tquesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, 'val2014')

saveLossFileName = 'outputFiles/jzs1_vgg_feats_%s_batch_%d_lr_%.4f.txt'%(quesTypes[0].replace(' ','-'), BATCH_SIZE, lr)
savePredFileName = 'outputFiles/jzs1_vgg_feats_%s_batch_%d_lr_%.4f_pred.txt'%(quesTypes[0].replace(' ','-'), BATCH_SIZE, lr)
saveModelFileName = 'models/jzs1_vgg_feats_%s_batch_%d_lr_%.4f.hdf5'%(quesTypes[0].replace(' ','-'), BATCH_SIZE, lr)

# Check if directory exists
if not os.path.exists('outputFiles'):
	os.makedirs('outputFiles')
if not os.path.exists('models'):
	os.makedirs('models')

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
#quesTypes = 'all'	# defined above

if quesTypes == 'all':
	annIdsA = vqa.getQuesIds()
	tannIdsA = tvqa.getQuesIds()
	imgIdsA = vqa.getImgIds()
	timgIdsA = tvqa.getImgIds()
else:
	annIdsA = vqa.getQuesIds(quesTypes = quesTypes)
	tannIdsA = tvqa.getQuesIds(quesTypes = quesTypes)
	imgIdsA = vqa.getImgIds(quesTypes = quesTypes)
	timgIdsA = tvqa.getImgIds(quesTypes = quesTypes)

annsA = vqa.loadQA(annIdsA)
tannsA = tvqa.loadQA(tannIdsA)

if len(annsA) > TR_LIMIT_SIZE:
	annsA[TR_LIMIT_SIZE:] = []
	imgIdsA[TR_LIMIT_SIZE:] = []

if len(tannsA) > VAL_LIMIT_SIZE:
	tannsA[VAL_LIMIT_SIZE:] = []
	timgIdsA[VAL_LIMIT_SIZE:] = []

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
print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))

##### Load VGG features
vggFeats = sio.loadmat(vggFeatDir + 'vgg_feats.mat')		
vggFeats = vggFeats['feats']				# contains (4096, 123287) numpy array
image_ids = open(vggFeatDir + 'coco_vgg_IDMap.txt').read().splitlines()
img_id_map = {}
for ids in image_ids:
	id_split = ids.split()
	img_id_map[id_split[0]] = int(id_split[1])	# contains {'coco_img_num': index}  cf.{'391895': 0}d



print('Build model ...')

### Image MLP (input: VGGNet features)
vggnet = Sequential()
vggnet.add(Dense(4096, HIDDEN_SIZE, activation="relu")) # to match the dimensions of image-question features (in Merge layer)

### Build Question RNN
qnet = Sequential()
qnet.add(Embedding(vocab_size+1, EMBED_SIZE, mask_zero=True))
qnet.add(JZS1(EMBED_SIZE, HIDDEN_SIZE))
qnet.add(Dense(HIDDEN_SIZE, HIDDEN_SIZE, activation="relu"))

### Merged model
model = Sequential()
model.add(Merge([vggnet, qnet], mode="concat", concat_axis=1)) # output_dim = 2*HIDDEN_SIZE
#model.add(Dense(2*HIDDEN_SIZE, HIDDEN_SIZE, activation="relu"))
model.add(RepeatVector(ans_maxlen))
model.add(JZS1(2*HIDDEN_SIZE, HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributedDense(HIDDEN_SIZE, vocab_size, activation="softmax")) # TimeDistributedDense

print('Model compiling ...')
#lr = 0.01	# defined at the top
opt = Adam(lr = lr)
model.compile(optimizer=opt, loss='categorical_crossentropy') # mean_squared_error, categorical_crossentropy


### Open file to save loss / accuracy
fResult = open(saveLossFileName, 'w')
fResult.write('Question type = %s\n'%(quesTypes))
fResult.write('BatchSize %d\n'%(BATCH_SIZE))
fResult.write('Epochs %d\n'%(EPOCHS))
fResult.write('VocabSize %d\n'%(vocab_size))
fResult.write('Learning rate %f\n'%(lr))
fResult.close()

### Open file to save predicted word sequence
fPredict = open(savePredFileName, 'w')
fPredict.write('Question type = %s\n'%(quesTypes))
fPredict.write('BatchSize %d\n'%(BATCH_SIZE))
fPredict.write('Epochs %d\n'%(EPOCHS))
fPredict.write('VocabSize %d\n'%(vocab_size))
fPredict.close()

class LossHistory(Callback):
	def on_batch_end(self, batch, logs={}):
		fResult = open(saveLossFileName, 'a+')
		loss = logs.get('loss')
		fResult.write('train %d %.4f\n'%(batch, loss))
		fResult.close()

class LossAccHistory(Callback):
	def on_epoch_end(self, epoch, logs={}):
		fPredict = open(savePredFileName, 'a+')
		pY = model.predict(tX, batch_size=BATCH_SIZE)
		ppY = np.zeros((len(pY), ans_maxlen, vocab_size), dtype=np.bool)
		for i in range(0, len(pY)):
			fPredict.write('\n%d %d'%(iEpoch, i))
			for j in range(0, ans_maxlen):
				index = taY[i, j]
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
			for j in range(0, ans_maxlen-1):
				if(twY[i, j, 0] == True & twY[i, j+1, 0] == True):
					nMask += 1
					dot = np.dot(ppY[i, j], tY[i, j])
					if dot == True:
						nacc += 1
		
		acc = float(nacc)/nMask
		fResult = open(saveLossFileName, 'a+')
		loss = logs.get('val_loss')
		# acc = logs.get('val_acc')
		fResult.write('val %d %.4f %.4f\n'%(iEpoch, loss, acc))
		fResult.close()
		
	def on_batch_end(self, batch, logs={}):
		fResult = open(saveLossFileName, 'a+')
		loss = logs.get('loss')
		fResult.write('train %d %.4f\n'%(batch, loss))
		fResult.close()

history = LossHistory()
acchistory = LossAccHistory()


##### Build Test matrices for text QA

print('Building Test matrices')
taX, taY = vectorize(test, word_idx, ques_maxlen, ans_maxlen)
tX_text, tY, tbY, twY = buildMat_text(taX, taY, ques_maxlen, ans_maxlen, vocab_size)
#data = Counter(tbY)
#taccmode = data.most_common(1).pop(0)[1]/float(len(taX))
tX_img = buildMat_feat(timgIdsA, vggFeats, img_id_map)
tX = [tX_img, taX]

begin = datetime.now()
numIter = int(np.ceil(len(annsA)/float(LIMIT_SIZE)))
iEpoch = 0
for i in range(EPOCHS):
	for j in range(numIter):		
		##### Build Train matrices for text QA
		print('Building Train matrices')
		aX, aY = vectorize(get_inputVec(vqa, annsA[j*LIMIT_SIZE:min((j+1)*LIMIT_SIZE,len(annsA)-1)]), word_idx, ques_maxlen, ans_maxlen)
		X_text, Y, bY, wY= buildMat_text(aX, aY, ques_maxlen, ans_maxlen, vocab_size)
		#data = Counter(bY)
		#accmode = data.most_common(1).pop(0)[1]/float(len(aX))
		X_img = buildMat_feat(imgIdsA[j*LIMIT_SIZE:min((j+1)*LIMIT_SIZE,len(annsA)-1)], vggFeats, img_id_map)
		#X = [X_img, aX]		# Use aX instead of X_text in training
		
		print('X_text.shape = {}'.format(X_text.shape))
		print('X_img.shape = {}'.format(X_img.shape))
		print('Y.shape = {}'.format(Y.shape))
		print('epoch = %d, index = %d'%(i+1, j+1))
		
		##### Training (+logging)
		print('Training ...')
		if j != (numIter-1):
			model.fit([X_img, aX], Y, batch_size=BATCH_SIZE, nb_epoch=1, verbose=1, show_accuracy=True, callbacks=[history], sample_weight=wY)
		else:
			model.fit([X_img, aX], Y, batch_size=BATCH_SIZE, nb_epoch=1, validation_data=(tX, tY, twY), verbose=1, show_accuracy=True, callbacks=[acchistory], sample_weight=wY)
	iEpoch += 1

end = datetime.now()
diff = end - begin
Sec = diff.total_seconds()
Min = int(Sec/60)
Hour = int(Min/60)
Day = int(Hour/24)
Sec -= 60*Min
Min -= 60*Hour
Hour -= 24*Day

model.save_weights(saveModelFileName, overwrite=True)

#print('X.shape = {}'.format(X.shape))
#print('Y.shape = {}'.format(Y.shape))
#print('ques_maxlen, ans_maxlen = {}, {}'.format(ques_maxlen, ans_maxlen))
#print('mode acc / test mode acc = %.4f / %.4f\n'%(accmode, taccmode))
print('Total learning time = %ddays %d:%d:%d\n\n'%(Day, Hour, Min, Sec))

fResult = open(saveLossFileName, 'a+')
#fResult.write('mode acc / test mode acc = %.4f / %.4f\n'%(accmode, taccmode))
fResult.write('Total learning time = %ddays %d:%d:%d\n\n'%(Day, Hour, Min, Sec))
fResult.close()
