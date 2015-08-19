from __future__ import division
import numpy
import re
import math
import operator
import numpy as np
from numpy import linalg as LA

numpy.set_printoptions(threshold=numpy.nan)

def cosSim(vector1, vector2, par = 'norm'):
	all_keys = set(vector1.keys() + vector2.keys())

	for key in all_keys:
		if key not in vector1:
			vector1[key] = 0
		if key not in vector2:
			vector2[key] = 0

	array1 = np.array(vector1.values())
	array2 = np.array(vector2.values())

	if par is 'bin':
		for item in range(0, len(array1)):
			if array1[item] > 0:
				array1[item] = 1
		for item in range(0, len(array2)):
			if array2[item] > 0:
				array2[item] = 1

	return np.dot(array1, array2)/(LA.norm(array1) * LA.norm(array2))

def sumOfList(list):
	sum_ = 0
	for item in list:
		sum_ += item
	return sum_

def sumOfElement(element, elements, frequencies):
	sum_ = 0
	for item in range(0, len(frequencies)):
		if elements[item] == element:
			sum_ += frequencies[item]
	return sum_

array = []

with open('target_context_count.sm') as raw:
# with open('ex01.sm') as raw:
	corpus = raw.read()
	lines = corpus.splitlines()
	for line in range(0, len(lines)):
		array.append(re.split(r'\t', lines[line]))

# print array
targets =  [i[0] for i in array]
contexts = [i[1] for i in array]
freqs =    [int(i[2]) for i in array]
registers = [i[3] for i in array]
ppmis =    []

sarcVector = {}
genuVector = {}

# print sumOfElement('think', contexts, freqs)
wc = sumOfList(freqs)
# print wc
for context in range(0, len(contexts)):
	# print sumOfElement('touser', contexts, freqs)/wc
	# print (freqs[context]/wc) / ((sumOfElement(contexts[context], contexts, freqs)/wc) * (sumOfElement(targets[context], targets, freqs)/wc))
	ppmi = math.log((freqs[context]/wc) / ((1/2) * (sumOfElement(contexts[context], contexts, freqs)/wc)), 2)
	if ppmi < 0:
		ppmi = 0
	# print registers[context]
	if(registers[context] == 'sarcastic'):
		sarcVector[contexts[context]] = ppmi
	elif(registers[context] == 'genuine'):
		genuVector[contexts[context]] = ppmi
	ppmis.append(ppmi)

sorted_sarcVector = sorted(sarcVector.items(), key=operator.itemgetter(1), reverse=True)
sorted_genuVector = sorted(genuVector.items(), key=operator.itemgetter(1), reverse=True)
# print sorted_sarcVector
# print '\n\n\n\n'
# print sorted_genuVector

# print sorted_sarcVector, '\n','\n', sorted_genuVector
print cosSim(sarcVector, genuVector, 'bin')

with open('ppmiOutput.txt', 'w') as raw:
	raw.write('{}\n{}\n\n\n{}\n{}'.format('Sarcastic', sorted_sarcVector, 'Genuine', sorted_genuVector))
