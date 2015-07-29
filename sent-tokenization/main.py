import operator
from collections import defaultdict
import string
import numpy as np
import scipy.sparse
import sys, getopt
import math

np.set_printoptions(threshold=np.nan)

stopWords = open('stop-words.txt').read().split()

def getLines(file):
	lines = []
	with open(file) as raw:
		sents = raw.read().lower().splitlines()
	for sent in range(0, len(sents)):
		exclude = set(string.punctuation)
		sents[sent] = ''.join(ch for ch in sents[sent] if ch not in exclude)
		lines.append(sents[sent].split())
	return lines

# WORD = 'now'

CORPUS = getLines('corpus.txt')

def mima(MINORMAX, val, given):
	if MINORMAX is 'min':
		if given < val:
			given = val
	if MINORMAX is 'max':
		if given > val:
			given = val 
	return given

def getContext(word, data, windowSize):
	# data = getLines(data_raw)
	contexts = defaultdict(lambda: 0)
	for line in range(0, len(data)):
		if word in data[line]:
			# print data[line]
			for i in range(mima('min', 0, data[line].index(word)-windowSize), mima('max', len(data[line]), data[line].index(word)+windowSize + 1)):
				# print data[line][i]
				if data[line][i] not in stopWords and data[line][i] != word:
					contexts[data[line][i]] += 1
	# return sorted(contexts.items(), key=operator.itemgetter(1), reverse=True)
	return contexts

def cosSim(data1, data2):
	keys1=np.unique(np.array((data1,data2)).T[0])
	keys2=np.unique(np.array((data1,data2)).T[1])
	
	all_keys=np.sort(np.append(keys1, keys2))

	# print all_keys

	array1=np.array([[i,data1.get(i,0)] for i in all_keys])
	array2=np.array([[i,data2.get(i,0)] for i in all_keys])

	# print array1, "\n", array2

	array1_i = np.array([i[1] for i in array1], dtype=float)
	array2_i = np.array([i[1] for i in array2], dtype=float)

	# print array1_i, "\n", array2_i

	file1 = open('serialized.txt', 'w+')

	file1.write('Word 1 Frequencies:\n' + np.array_repr(array1) + '\nVector:\n' +  np.array_repr(array1_i))
	file1.write('\n\nWord 2 Frequencies:\n' + np.array_repr(array2) + '\nVector:\n' +  np.array_repr(array2_i))

	file1.close()

	return (np.dot(array1_i, array2_i))/(np.linalg.norm(array1_i)*np.linalg.norm(array2_i))


def wordCount(word, data):
	tmp = 0
	for line in range(0, len(data)):
		for i in range(0, len(data[line])):
			if word == '\a':
				tmp += 1
			elif data[line][i] == word:
				tmp += 1
	return tmp

def ppmi(data1, data2, word1, word2):
	wc = wordCount('\a', CORPUS)
	# print wc
	keys1=np.unique(np.array((data1,data2)).T[0])
	keys2=np.unique(np.array((data1,data2)).T[1])
	
	all_keys=np.sort(np.append(keys1, keys2))

	# print all_keys

	array1=np.array([[i,data1.get(i,0)] for i in all_keys])
	array2=np.array([[i,data2.get(i,0)] for i in all_keys])

	# print array1, "\n", array2

	array1_i = np.array([i[1] for i in array1], dtype=float)
	array2_i = np.array([i[1] for i in array2], dtype=float)

	# print array2_i[np.searchsorted(all_keys, word1)]
	print array1

	# print array2_i[np.searchsorted(all_keys, word1)], "/", wc
	# print wordCount(word1, CORPUS), "/", wc
	# print wordCount(word2, CORPUS), "/", wc

	print array2_i[np.searchsorted(all_keys, word1)]*wc
	print ((wordCount(word1, CORPUS))*(wordCount(word2, CORPUS)))
	# print wordCount(word1, CORPUS)/
	print wc
	out = math.log((array2_i[np.searchsorted(all_keys, word1)]*wc)/((wordCount(word1, CORPUS))*(wordCount(word2, CORPUS))), 2)

	if out < 0:
		return 0
	return out

def main(argv):

	WINDOW_SIZE = 4

	word1 = "man"
	word2 = "business"

	data1 = getContext(word1, CORPUS, WINDOW_SIZE)
	data2 = getContext(word2, CORPUS, WINDOW_SIZE)
	
	# print cosSim(data1, data2)
	print ppmi(data1, data2, word1, word2)
	# print data1
	# print wordCount('\a', CORPUS)

if __name__ == "__main__":
	main(sys.argv[1:])