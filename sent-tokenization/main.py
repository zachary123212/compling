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
	"""
	Args:
		file: a corpus txt file with line breaks between sentences

	Returns: 
		a list of lists, each 1st level being sentences and 2nd-words, with removed stop words and all lowercase.
		Example:
			[['lorem', 'ipsum', 'dolor'], ['sit', 'consectetur', 'adipiscing'], ['elit']]
	"""
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
	"""
	Args:
		MINORMAX: determines whether function defines minimum or maximum
		val: defines maximum or minimum value for given
		given: input value to be compared with val

	Returns:
		given, constrained by minimum or maximum val
	"""
	if MINORMAX is 'min':
		if given < val:
			given = val
	if MINORMAX is 'max':
		if given > val:
			given = val 
	return given

def getContext(word, data, windowSize):
	"""
	Args:
		word: input word whose context is returned 
		data: corpus from which context of word is retrieved
		windowSize: window size (left and right) for obtaining of word context

	Returns:
		contexts, a default dictionary with default value 0, key: word, item: frequency of word in windowSize of data corpus
	"""
	# data = getLines(data_raw)
	contexts = defaultdict(lambda: 0)
	for line in range(0, len(data)):
		if word in data[line]:
			# print data[line]
			for i in range(mima('min', 0, data[line].index(word)-windowSize), mima('max', len(data[line]), data[line].index(word)+windowSize + 1)):
				# print data[line][i]
				if data[line][i] not in stopWords and data[line][i] != word:
					# contexts[data[line][i]] += 1
					contexts[data[line][i]] += 1
	# return sorted(contexts.items(), key=operator.itemgetter(1), reverse=True)
	return contexts

def cosSim(contexts1, contexts2):
	"""
	Args:
		contexts1: contexts of word to be compared with contexts of another word
		contexts2: contexts of word to be compared with contexts of another word

	Returns:
		cosine distributional probability of two words
		also outputs contexts arrays to serialized.txt
	"""
	keys1=np.unique(np.array((contexts1,contexts2)).T[0])
	keys2=np.unique(np.array((contexts1,contexts2)).T[1])

	# all_keys=None

	# for key in range(0, len(keys2)):
	# 	if keys2[key] not in all_keys:
	# 		np.append(all_keys, keys2)

	all_keys=np.sort(np.unique(np.append(keys1, keys2)))

	# print all_keys

	array1=np.array([[i,contexts1.get(i,0)] for i in all_keys])
	array2=np.array([[i,contexts2.get(i,0)] for i in all_keys])

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
	"""
	Args:
		word: word whose frequency is to be returned
		data: corpus in which word's frequency is gaged

	Returns: 
		if word is '\a', returns number of words in data
		otherwise, returns frequency of word in data
	"""
	tmp = 0
	for line in range(0, len(data)):
		for i in range(0, len(data[line])):
			if word == '\a':
				tmp += 1
			elif data[line][i] == word:
				tmp += 1
	return tmp

def ppmi(contexts1, contexts2, word1, word2):
	"""
	Args:
		contexts1: context dictionary of word 1
		contexts2: context dictionary of word 2
		word1: word to be compared
		word2: word to be compared

	Returns:
		PPMI of word1, given word2
	"""
	wc = wordCount('\a', CORPUS)
	# print wc
	keys1=np.unique(np.array((contexts1,contexts2)).T[0])
	keys2=np.unique(np.array((contexts1,contexts2)).T[1])
	
	all_keys=np.sort(np.append(keys1, keys2))

	# print all_keys

	array1=np.array([[i,contexts1.get(i,0)] for i in all_keys])
	array2=np.array([[i,contexts2.get(i,0)] for i in all_keys])

	# print array1, "\n", array2

	array1_i = np.array([i[1] for i in array1], dtype=float)
	array2_i = np.array([i[1] for i in array2], dtype=float)

	# print array2_i[np.searchsorted(all_keys, word1)]
	# print array1

	# print array2_i[np.searchsorted(all_keys, word1)], "/", wc
	# print wordCount(word1, CORPUS), "/", wc
	# print wordCount(word2, CORPUS), "/", wc

	# print array2

	# print array2_i[np.searchsorted(all_keys, word1)]*wc
	# print ((wordCount(word1, CORPUS))*(wordCount(word2, CORPUS)))
	# print wordCount(word1, CORPUS)
	# print wc

	if array2_i[np.searchsorted(all_keys, word1)]*wc == 0:
		return 0

	out = math.log((array2_i[np.searchsorted(all_keys, word1)]*wc)/((wordCount(word1, CORPUS))*(wordCount(word2, CORPUS))), 2)

	if out < 0:
		return 0
	return out

def main(argv):

	WINDOW_SIZE = 5

	word1 = "dog"
	word2 = "man"

	contexts1 = getContext(word1, CORPUS, WINDOW_SIZE)
	contexts2 = getContext(word2, CORPUS, WINDOW_SIZE)
	
	print cosSim(contexts1, contexts2)
	# print ppmi(contexts1, contexts2, word1, word2)
	# print contexts1
	# print wordCount('\a', CORPUS)

if __name__ == "__main__":
	main(sys.argv[1:])