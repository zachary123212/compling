from collections import defaultdict
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
import re
from nltk.corpus import stopwords

text = open('corpus.txt').read().decode('utf8').lower()
# text = re.sub('\'s', '', text)

tokens = RegexpTokenizer(r'\w+').tokenize(text)
targets = word_tokenize(open('targets.txt').read())

k = 10

wordDict = defaultdict(list)

filtered_words = [word for word in tokens if word not in stopwords.words('english')]

for word in range(0, len(filtered_words)):
	if filtered_words[word] in targets:
		for i in range(max(0, word-k), word+k+1):
			print filtered_words[i],
		print "\n"
