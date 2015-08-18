import operator
from collections import defaultdict
import string
import numpy as np
import scipy.sparse
import sys, getopt
import math
from progressbar import ProgressBar
import os

np.set_printoptions(threshold=np.nan)

stopWords = open('stop-words.txt').read().split()

WINDOW_SIZE = 10

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
		words = sents[sent].split()
		wordsR = []
		for word in words:
			if word not in stopWords:
				wordsR.append(word)
		lines.append(wordsR)
	# print lines
	return lines

# WORD = 'now'

def sumOfCol(array):
	val = 0
	for row in range(0, len(array)):
		val += array[row]
	return val
def sumOfElement(element, arrayS, arrayN):
	val = 0
	for row in range(0, len(arrayS)):
		if arrayS[row] is element:
			val += arrayN[row]
	return val

CORPUS = getLines('corpora/output.txt')
# CORPUS = getLines('corpus.txt')

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
def getContext(word, data, windowSize, par='default'):
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
					if par == 'default':
						contexts[data[line][i]] += 1
					elif par == 'bool':
						contexts[data[line][i]] = 1
	# return sorted(contexts.items(), key=operator.itemgetter(1), reverse=True)
	return contexts
def cosSim(word1, word2, par):
	"""
	Args:
		contexts1: contexts of word to be compared with contexts of another word
		contexts2: contexts of word to be compared with contexts of another word

	Returns:
		cosine distributional probability of two words
		also outputs contexts arrays to serialized.txt
	"""

	contexts1 = getContext(word1, CORPUS, WINDOW_SIZE)
	contexts2 = getContext(word2, CORPUS, WINDOW_SIZE)

	keys1=np.unique(np.array((contexts1,contexts2)).T[0])
	keys2=np.unique(np.array((contexts1,contexts2)).T[1])

	# all_keys=None

	# for key in range(0, len(keys2)):
	# 	if keys2[key] not in all_keys:
	# 		np.append(all_keys, keys2)

	all_keys=np.sort(np.unique(np.append(keys1, keys2)))

	# print all_keys
	if par is 'ppmi':
		array1=np.array([[i,ppmi(getContext(i, CORPUS, WINDOW_SIZE), contexts1, i, word1)] for i in all_keys])
		array2=np.array([[i,ppmi(getContext(i, CORPUS, WINDOW_SIZE), contexts2, i, word2)] for i in all_keys])
	elif par is 'freq':
		array1=np.array([[i,contexts1.get(i,0)] for i in all_keys])
		array2=np.array([[i,contexts2.get(i,0)] for i in all_keys])
	elif par is 'bin':
		array1=np.array([[i,contexts1.get(i,0)] for i in all_keys])
		for i in range(0, len(array1)):
			if array1[i][1].astype(int) > 1:
				array1[i][1] = 1
		array2=np.array([[i,contexts2.get(i,0)] for i in all_keys])
		for i in range(0, len(array2)):
			if array2[i][1].astype(int) > 1:
				array2[i][1] = 1
	else:
		return 'ERROR'

	# print contexts1, "\n", contexts2
	# print array1, "\n", array2

	array1_i = np.array([i[1] for i in array1], dtype=float)
	array2_i = np.array([i[1] for i in array2], dtype=float)

	# print array1_i, "\n", array2_i

	out = (np.dot(array1_i, array2_i))/(np.linalg.norm(array1_i)*np.linalg.norm(array2_i))

	file1 = open('serialized.txt', 'w+')

	file1.write('Word 1 Frequencies:\n' + np.array_repr(array1) + '\nVector:\n' +  np.array_repr(array1_i))
	file1.write('\n\nWord 2 Frequencies:\n' + np.array_repr(array2) + '\nVector:\n' +  np.array_repr(array2_i))
	file1.write('\n\n' + np.array_repr(out))

	file1.close()

	return out
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
	# wc = sumOfCol()
	# print wc
	keys1=np.unique(np.array((contexts1,contexts2)).T[0])
	keys2=np.unique(np.array((contexts1,contexts2)).T[1])

	all_keys=np.sort(np.unique(np.append(keys1, keys2)))

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

	# print array2_i[np.searchsorted(all_keys, word1)]
	# print ((wordCount(word1, CORPUS))*(wordCount(word2, CORPUS)))
	# print wordCount(word1, CORPUS)
	# print wc

	wc = sumOfCol(array1_i)

	num = np.nonzero(all_keys == word1)[0]
	if num.size == 0:
		return 0

	if array2_i[np.searchsorted(all_keys, word1)] == 0:
		return 0

	# print array2_i[num]
	# print array2_i[num]/((sumOfCol(array2_i)/(sumOfCol(array1_i)+sumOfCol(array2_i))) * (sumOfCol(array1_i)/(sumOfCol(array1_i)+sumOfCol(array2_i))))

	out = math.log( (array2_i[num])/((sumOfCol(array2_i)/(sumOfCol(array1_i)+sumOfCol(array2_i))) * (sumOfCol(array1_i)/(sumOfCol(array1_i)+sumOfCol(array2_i)))) ,2)

	# out = math.log((array2_i[num]*wc/((sumOfElement(word1, all_keys, array1_i)+sumOfElement(word1, all_keys, array2_i))*(sumOfElement(word2, all_keys, array2_i)+sumOfElement(word2, all_keys, array_i)))), 2)

	# out = math.log((array2_i[num]*wc)/((wordCount(word1, CORPUS))*(wordCount(word2, CORPUS))), 2)

	if out < 0:
		return 0
	return out

def formalize(sarc):
	all_words = defaultdict(lambda: 0)
	for line in range(0, len(CORPUS)):
		for word in range(0, len(CORPUS[line])):
			all_words[CORPUS[line][word]] += 1
	all_words_sorted = sorted(all_words.items(), key=operator.itemgetter(1), reverse=True)
	# print all_words_sorted
	with open('contexts.cols', 'w') as raw:
		for word in range(0, len(all_words_sorted)):
			# print all_words_sorted[word][0]
			raw.write(all_words_sorted[word][0] + '\n')

	with open('targets.rows', 'w') as raw:
		for word in range(0, 1):
			raw.write(all_words_sorted[word][0] + '\n')

	with open('target_context_count.sm', 'a') as raw:
		# raw.write('target_word\tcontext_word\tcount\n')
		for target in range(0, 1):
			pbar = ProgressBar(maxval = len(all_words_sorted)).start()
			for context in range(0, len(all_words_sorted)):
				# print all_words_sorted[target]
				pbar.update(context+1)
				context_ = getContext(all_words_sorted[target][0], CORPUS, WINDOW_SIZE)[all_words_sorted[context][0]]
				if context_ > 2:
					# print all_words_sorted[context]
					# raw.write('{}\t{}\t{}\t{}\n'.format(all_words_sorted[target][0],all_words_sorted[context][0],context_, ppmi(getContext(all_words_sorted[context][0], CORPUS, WINDOW_SIZE), getContext(all_words_sorted[target][0], CORPUS, WINDOW_SIZE), all_words_sorted[context][0], all_words_sorted[target][0])))
					raw.write('{}\t{}\t{}\t{}\n'.format(all_words_sorted[target][0],all_words_sorted[context][0], context_, sarc))

				# print('\n')
			pbar.finish()

def printCosSim(words):
	with open('cosSim.txt', 'w') as raw:
		raw.write('word1\tword2\tfreq\tbin\tppmi\n')
		# pbar = ProgressBar().start()
		for wordT in words:
			for wordC in words:
				# pbar.update(wordC.T+1)
				raw.write('{}\t{}\t{}\t{}\t{}\n'.format(wordT, wordC, cosSim(wordT, wordC, 'freq'), cosSim(wordT, wordC, 'bin'), cosSim(wordT, wordC, 'ppmi')))
				# raw.write('{}\t{}\t{}\t{}\t{}\n'.format('wordT', 'wordC', 'cosSim(wordT, wordC,)', 'cosSim(wordT, wordC,)', 'cosSim(wordT, wordC,)'))
		# pbar.finish()


def main(argv):

	# word1 = "man"
	# word2 = "woman"
	# print ppmi(getContext(word1, CORPUS, WINDOW_SIZE), getContext(word2, CORPUS, WINDOW_SIZE), word1, word2)

	# x = np.array([1, 3, 1, 3, 2, 93, 32, 32])
	# print np.nditer(x, 1)

	# print ppmi(getContext('cat', CORPUS, WINDOW_SIZE), getContext('among', CORPUS, WINDOW_SIZE), 'cat', 'among')

	# print cosSim(word1, word2, 'ppmi')

	# printCosSim(['lovely', 'touser'])

	formalize(sys.argv[1])

	# print ppmi(getContext('', CORPUS, WINDOW_SIZE), getContext('dog', CORPUS, WINDOW_SIZE), 'cat', 'dog')

	# print ppmi(contexts1, contexts2, word1, word2)
	# print contexts1
	# print wordCount('\a', CORPUS)

if __name__ == "__main__":
	main(sys.argv[1:])