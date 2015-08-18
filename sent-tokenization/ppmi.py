from __future__ import division
import numpy
import re
import math

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
ppmis =    []

# print sumOfElement('think', contexts, freqs)
wc = sumOfList(freqs)
print wc
for context in range(0, len(contexts)):
	# print (freqs[context]/wc) / ((sumOfElement(contexts[context], contexts, freqs)/wc) * (sumOfElement(targets[context], targets, freqs)/wc))
	ppmi = math.log((freqs[context]/wc) / ((sumOfElement(contexts[context], contexts, freqs)/wc) * (sumOfElement(targets[context], targets, freqs)/wc)), 2)
	if ppmi < 0:
		ppmi = 0
	ppmis.append(ppmi)
print ppmis