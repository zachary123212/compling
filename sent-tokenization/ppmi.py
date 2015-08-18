from __future__ import division
import numpy
import re
import math
import operator

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

with open('ppmiOutput.txt', 'w') as raw:
	raw.write('{}\n{}\n\n\n{}\n{}'.format('Sarcastic', sorted_sarcVector, 'Genuine', sorted_genuVector))