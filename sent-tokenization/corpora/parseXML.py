import xml.etree.ElementTree as ET

CORPUS='tweet.txt.xml'

tree = ET.parse(CORPUS)
root = tree.getroot()

with open('output.txt', 'w') as raw:
	for sentence in root.iter('sentence'):
		for lemma in sentence.iter('lemma'):
			raw.write(lemma.text+' ')
		raw.write('\n')