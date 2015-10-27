# from __future__ import absolute_import
# from __future__ import print_function
import random
import re
import numpy as np
import matplotlib.pyplot as plt

f = open('jzs1_embed_den_relu_time_soft_pre_adam_cat.txt', 'r')
batch = 0
epoch = 0
for line in f:
	if 'BatchSize' in line:
		batch = float(line.split(' ')[1])
	elif 'Epochs' in line:
		epoch = float(line.split(' ')[1])
		break

INIT = 10000
FONT_SIZE = 25
yloss = []
vloss = []
vacc = []
x = INIT
xv = []
for line in f:
	if 'train' in line:
		yloss.append(float(line.split(' ')[2]))
		x += 1
	elif 'val' in line:
		vloss.append(float(line.split(' ')[2]))
		vacc.append(float(line.split(' ')[3]))
		xv.append(x)


f.close()
x = range(INIT, x)

plt.figure(1)
plt.subplot(121)
line_train_loss = plt.plot(x, yloss, label='train_loss', color='blue')
line_val_loss = plt.plot(xv, vloss, label='val_loss', linewidth=2, color='red')
plt.annotate('train={}'.format(yloss[0]), xy=(INIT, yloss[0]), fontsize=FONT_SIZE)
plt.annotate('train={}'.format(yloss[len(yloss)-1]), xy=(len(yloss)-1, yloss[len(yloss)-1]), fontsize=FONT_SIZE)
plt.annotate('val={}'.format(vloss[0]), xy=(INIT, vloss[0]), fontsize=FONT_SIZE)
plt.annotate('val={}'.format(vloss[len(vloss)-1]), xy=(len(yloss)-1, vloss[len(vloss)-1]), fontsize=FONT_SIZE)
plt.legend(loc='upper right')

plt.subplot(122)
#line_train_acc = plt.plot(x, vacc, label='train_acc', color='blue')
line_val_acc = plt.plot(xv, vacc, label='val_acc', linewidth=2, color='green')
#plt.annotate('train={}'.format(yacc[0]), xy=(INIT, yacc[0]), fontsize=FONT_SIZE)
#plt.annotate('train={}'.format(yacc[len(yacc)-1]), xy=(len(yacc)-1, yacc[len(yacc)-1]), fontsize=FONT_SIZE)
plt.annotate('val={}'.format(vacc[0]), xy=(INIT, vacc[0]), fontsize=FONT_SIZE)
plt.annotate('val={}'.format(vacc[len(vacc)-1]), xy=(len(yloss)-1, vacc[len(vacc)-1]), fontsize=FONT_SIZE)
plt.legend(loc='upper right')

plt.show()
