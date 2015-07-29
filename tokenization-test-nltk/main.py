import nltk
import re
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize

swords = stopwords.words('english')
aliceRaw = gutenberg.raw('carroll-alice.txt')

aliceRaw = aliceRaw.encode('utf8').lower()
alice = word_tokenize(aliceRaw)

for i in range(0, len(alice)):
	if alice[i] in swords or alice[i] == ',':
		alice[i] = "delete-me"

#for i in range(0, len(aliceTemp)):
#	if aliceTemp[i] == "delete-me":
#		alice.remove(alice[i])

alice[:] = [item for item in alice if item != "delete-me"]

freqs = nltk.FreqDist(alice)
alice_sorted = sorted(alice, key=lambda x: freqs[x.lower()], reverse=True)

print alice_sorted