from __future__ import absolute_import
from __future__ import print_function

from functools import reduce
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import re
import cv2

END_MARK = 1

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

def vectorize(data, word_idx, ques_maxlen, ans_maxlen, pad='post'):
	rX = []
	rY = []
	for ques, ans in data:
		x = [word_idx[w] for w in ques]
		y = [word_idx[w] for w in ans]
		rX.append(x)
		rY.append(y+[END_MARK])
	return pad_sequences(rX, maxlen=ques_maxlen), pad_sequences(rY, maxlen=ans_maxlen, padding=pad)

def buildMat_text(aX, aY, ques_maxlen, ans_maxlen, vocab_size, post_padding=True):
	X = np.zeros((len(aX), ques_maxlen, vocab_size), dtype=np.bool)
	Y = np.zeros((len(aY), ans_maxlen, vocab_size), dtype=np.bool)
	bY = []
	wY = np.zeros((len(aY), ans_maxlen, 1), dtype=np.bool)
	for i in range(0, len(aX)):
		start = False
		for j in range(0, ques_maxlen):
			if start == False:
				if aX[i, j] != 0:
					start = True
			if start == True:
				X[i, j, aX[i, j]-1] = 1
	
	if post_padding == True:
		for i in range(0, len(aY)):
			code = 0
			count = 0
			for j in range(0, ans_maxlen):
				if aY[i, j] == 0:
					bY.append(code)
					break
				else:
					wY[i, j, 0] = True
					Y[i, j, aY[i, j]-1] = 1
					code += (aY[i, j] * (vocab_size ** (count)))
					count += 1
	else:
		for i in range(0, len(aY)):
			start = False
			code = 0
			count = 0
			for j in range(0, ans_maxlen):
				if aY[i, j] != 0:
					wY[i, j, 0] = True
				if start == False:
					if aY[i, j] != 0:
						start = True
				if start == True:
					Y[i, j, aY[i, j]-1] = 1
					code += (aY[i, j] * (vocab_size ** (count)))
					count += 1
			bY.append(code)
	
	for i in range(0, len(aY)):
		start = False
		code = 0
		count = 0
		for j in range(0, ans_maxlen):
			if aY[i,j] != 0:
				wY[i,j,0] = True
			if start == False:
				if aY[i, j] != 0:
					start = True
			if start == True:
				Y[i, j, aY[i, j]-1] = 1
				code += (aY[i, j] * (vocab_size ** (count)))
				count += 1
		bY.append(code)
	return X, Y, bY, wY

def buildMat_img(imgIds, imgDir, dataType):
	X = np.zeros(shape=(len(imgIds),3,224,224))
	for i in xrange(len(imgIds)):
		imgFilename = 'COCO_' + dataType + '_' + str(imgIds[i]).zfill(12) + '.jpg'
		img = cv2.resize( cv2.imread(imgDir + imgFilename), (224,224) )
		img = img.transpose((2,0,1))
		X[i] = np.expand_dims(img, axis=0 )
	return X

def buildMat_feat(imgIds, vggFeats, id_map):
	X = np.zeros(shape=(len(imgIds), vggFeats.shape[0]))
	for i in xrange(len(imgIds)):
		X[i] = vggFeats[:,id_map[str(imgIds[i])]]
	return X
