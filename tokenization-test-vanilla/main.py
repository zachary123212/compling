import re
import operator
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab

file = open('corpus.txt')

sample = file.read()
sample = sample.lower()
data = re.compile('\S+')

tokens = {}
wordNum = 0

for i in data.findall(sample):
	if i in tokens:
		tokens[i] += 1
	else:
		tokens[i] = 1
		wordNum += 1
	if i in open('stop-words.txt').read():
		tokens[i] = 0

tokens_sorted = sorted(tokens.items(), key=operator.itemgetter(1), reverse=True)

for i in range(0, 4)
	for word in tokens_so 

# frequency = []
# word = []

# print tokens_sorted

# for i in tokens_sorted:
# 	frequency.append(i[1])
# 	word.append(i[0])

# (mu, sigma) = norm.fit(frequency)

# print mu, " ", sigma

# fig = plt.figure()
# ax = fig.add_subplot(111)

# plt.hist(frequency, len(word), bottom=1, color='blue')
# ax.set_yscale('log')
# plt.show()


