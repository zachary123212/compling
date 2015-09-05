from __future__ import division
import numpy
import re
import math
import operator
import numpy as np
from numpy import linalg as LA

numpy.set_printoptions(threshold=numpy.nan)

def wordCount(inWord, corpus):
	count = 0
	for line in corpus:
		for word in line:
			if word == inWord:
				count += 1
	return count

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]

def cosSim(vector1, vector2, par = 'norm'):
	values1 = []
	values2 = []
	all_keys = f7((vector1.keys() + vector2.keys()))
	# print all_keys
	for key in all_keys:
		if key not in vector1:
			values1.append(0)
		else:
			values1.append(vector1[key])

		if key not in vector2:
			values2.append(0)
		else:
			values2.append(vector2[key])



	array1 = np.array(values1)
	array2 = np.array(values2)

	if par is 'bin':
		for item in range(0, len(array1)):
			if array1[item] > 0:
				array1[item] = 1
		for item in range(0, len(array2)):
			if array2[item] > 0:
				array2[item] = 1

	# print array1, '\n\n\n'
	# print array2, '\n\n\n'

	# print np.dot(array1, array2)/(LA.norm(array1) * LA.norm(array2))
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

with open('target_context_count_trained.sm') as raw:
# with open('ex01.sm') as raw:
	corpus = raw.read()
	lines = corpus.splitlines()
	for line in range(0, len(lines)):
		array.append(re.split(r'\t', lines[line]))

# print array
targets =   [i[0] for i in array]

sarcVectors = {}
genuVectors = {}

sarcVectorsF = {}
genuVectorsF = {}

# print sumOfElement('think', contexts, freqs)
# print wc
for target in set(targets):
	# print array
	targetArray = [i for i in array if i[0] == target]
	# print targetArray
	contexts =  [i[1] for i in targetArray]
	freqs = 	[int(i[2])/wordCount(i[1], array) for i in targetArray]
	registers = [i[3] for i in targetArray]
	# print registers
	ppmis = []

	wc = sumOfList(freqs)
	sarcVectors[target] = {}
	genuVectors[target] = {}

	sarcVectorsF[target] = {}
	genuVectorsF[target] = {}
	# print contexts
	for context in range(0, len(contexts)):
		# print sumOfElement('touser', contexts, freqs)/wc
		# print (freqs[context]/wc) / ((sumOfElement(contexts[context], contexts, freqs)/wc) * (sumOfElement(targets[context], targets, freqs)/wc))
		ppmi = math.log((freqs[context]/wc) / ((1/2) * (sumOfElement(contexts[context], contexts, freqs)/wc)), 2)
		if ppmi < 0:
			ppmi = 0
		
		if(registers[context] == 'sarcastic'):
			sarcVectors[target][contexts[context]] = ppmi
			sarcVectorsF[target] = {contexts[i]: freqs[i] for i in range(0, len(targetArray)) if registers[i] == "sarcastic"}
			# print sarcVectorsF[target]
		elif(registers[context] == 'genuine'):
			genuVectors[target][contexts[context]] = ppmi
			genuVectorsF[target] = {contexts[i]: freqs[i] for i in range(0, len(targetArray)) if registers[i] == "genuine"}

# sorted_sarcVector = sorted(sarcVector.items(), key=operator.itemgetter(1), reverse=True)
# sorted_genuVector = sorted(genuVector.items(), key=operator.itemgetter(1), reverse=True)
# print sorted_sarcVector
# print '\n\n\n\n'
# print sorted_genuVector

# print sorted_sarcVector, '\n','\n', sorted_genuVector
# print cosSim(sarcVector, genuVector, 'bin')



"""
with open('output.txt', 'w') as raw:
	raw.write('target1\ttarget2\tcosine_rawcount\tcosine_ppmi\tcosine_binary\n')
	for target in sarcVectors:
		for context in sarcVectors:
			# print target, ' ', context, ' ', cosSim(sarcVectors[target], sarcVectors[context])
			# print sarcVectors['fantastic']
			# print sarcVectors['fantastic'], '\n\n', sarcVectors['sweet']
			# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
			# print cosSim(sarcVectors['sweet'], sarcVectors['fantastic'])
			raw.write('{}_sarcastic\t{}_sarcastic\t{}\t{}\t{}\n'.format(target, context, cosSim(sarcVectorsF[target], sarcVectorsF[context]), cosSim(sarcVectors[target], sarcVectors[context]), cosSim(sarcVectorsF[target], sarcVectorsF[context], 'bin')))
"""



	# for target in genuVectors:
	# 	for context in genuVectors:
	# 		raw.write('{}_genuine\t{}_genuine\t{}\t{}\t{}\n'.format(target, context, cosSim(genuVectorsF[target], genuVectorsF[context]), cosSim(genuVectors[target], genuVectors[context]), cosSim(genuVectors[target], genuVectors[context], 'bin')))


input_ = ''
# while input_ is not '-1':
# 	input_ = raw_input('input sample data: ')
# 	inTarget = raw_input('input target word: ')
# 	inFreqs = {}
# 	for word in re.split(' ',input_):
# 		if word in inFreqs:
# 			inFreqs[word] += 1
# 		else:
# 			inFreqs[word] = 1
# 	# print inFreqs
# 	# print sarcVectorsF[inTarget]
# 	# print genuVectorsF[inTarget]
# 	print 'Sarcastic similarity: ', cosSim(inFreqs, sarcVectorsF[inTarget])
# 	print 'Literal similarity: ', cosSim(inFreqs, genuVectorsF[inTarget])
# 	if cosSim(inFreqs, sarcVectorsF[inTarget]) > cosSim(inFreqs, genuVectorsF[inTarget]):
# 		print 'Sarcastic'
# 	else:
# 		print 'Genuine'

with open('corpora/outputStatic.txt') as raw:
	data = raw.read()
	lines = data.splitlines()
	inData = []
	inFreqs = {}
	inAcc = {}
	answers = {}
	targets = []
	num = 0
	den = 0

	for line in range(0, len(lines)):
		inData.append(re.split(r'\t', lines[line]))

	for i in range(0, len(inData)):
		if inData[i][0] not in targets:
			targets.append(inData[i][0])
			answers[inData[i][0]] = [0,0,0,0,0]

	for item in inData:
		inFreqs[item[1]] = {}
		for word in re.split(' ',item[1]):
			if word in inFreqs[item[1]]:
				inFreqs[item[1]][word] += 1
			else:
				inFreqs[item[1]][word] = 1

		sarcSim = cosSim(inFreqs[item[1]], sarcVectorsF[item[0]], 'bin')
		genuSim = cosSim(inFreqs[item[1]], genuVectorsF[item[0]], 'bin')

		if sarcSim > genuSim:
			ans = 'sarcastic'
		elif genuSim > sarcSim:
			ans = 'genuine'
		else:
			ans = 'equal'
		
		if item[2] == 'sarcastic' and ans == 'sarcastic':
			answers[item[0]][0] += 1	
		elif item[2] == 'genuine' and ans == 'sarcastic':
			answers[item[0]][1] += 1	
		elif item[2] == 'sarcastic' and ans == 'genuine':
			answers[item[0]][2] += 1	
		elif item[2] == 'genuine' and ans == 'genuine':
			answers[item[0]][3] += 1	
		# if item[2] == ans:
		# 	answers[item[0]][0] += 1
		# answers[item[0]][1] += 1

	for answer in answers.items():
		answer[1][4] = (answer[1][0] + answer[1][3])/(answer[1][0] + answer[1][1] + answer[1][2] +answer[1][3])*100



# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
# print cosSim(sarcVectors['fantastic'], sarcVectors['sweet'])
# print cosSim(sarcVectors['sweet'], sarcVectors['fantastic'])

# print sarcVectors, '\n\n', sarcVectorsF

# with open('ppmiOutput.txt', 'w') as raw:
	# raw.write('{}\n{}\n\n\n{}\n{}'.format('Sarcastic', sorted_sarcVector, 'Genuine', sorted_genuVector))
