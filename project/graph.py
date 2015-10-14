# from __future__ import absolute_import
# from __future__ import print_function
import random
import re
import numpy as np
import matplotlib.pyplot as plt

f = open('lstm1_den_soft_post_adam_cat.txt', 'r')
y = []
x = 0
for line in f:
	if 'train' in line:
		y.append(float(line.split(' ')[3]))
		x += 1

f.close()
x = range(x)

plt.plot(x,y)
plt.show()
