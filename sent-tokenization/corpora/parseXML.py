import xml.etree.ElementTree as ET
import re

CORPUS='tweet.txt.xml'

tree = ET.parse(CORPUS)
root = tree.getroot()

with open('output.txt', 'w') as raw:
	for sentence in root.iter('sentence'):
		for token in sentence.iter('token'):
			if int(token.attrib['id']) > 2:
				raw.write(token.find('lemma').text+' ')
		raw.write('\n')