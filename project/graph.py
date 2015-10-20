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

INIT = 1000
y = []
vloss = []
vacc = []
x = INIT
xv = []
for line in f:
	if 'train' in line:
		y.append(float(line.split(' ')[2]))
		x += 1
	elif 'val' in line:
		vloss.append(float(line.split(' ')[2]))
		vacc.append(float(line.split(' ')[3]))
		xv.append(x)


f.close()
x = range(INIT, x)

line_train_loss = plt.plot(x, y, label='train_loss', color='blue')
line_val_loss = plt.plot(xv, vloss, label='val_loss', linewidth=2, color='red')
line_val_acc = plt.plot(xv, vacc, label='val_acc', linewidth=2, color='green')

plt.annotate('{}'.format(y[0]), xy=(INIT, y[0]), fontsize=15)
plt.annotate('{}'.format(y[len(y)-1]), xy=(len(y)-1, y[len(y)-1]), fontsize=15)
plt.annotate('{}'.format(vloss[0]), xy=(INIT, vloss[0]), fontsize=15)
plt.annotate('{}'.format(vloss[len(vloss)-1]), xy=(len(y)-1, vloss[len(vloss)-1]), fontsize=15)
plt.annotate('{}'.format(vacc[0]), xy=(INIT, vacc[0]), fontsize=15)
plt.annotate('{}'.format(vacc[len(vacc)-1]), xy=(len(y)-1, vacc[len(vacc)-1]), fontsize=15)


plt.legend(loc='upper right')

plt.show()
