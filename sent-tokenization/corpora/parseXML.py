import xml.etree.ElementTree as ET
import re
import os
import sys, getopt

CORPUS='tweet.txt.xml'

tree = ET.parse(CORPUS)
root = tree.getroot()

def main(argv):
	with open('output.txt', 'a') as raw:
		for sentence in root.iter('sentence'):
			raw.write('{}\t'.format(sys.argv[1]))
			for token in sentence.iter('token'):
				if int(token.attrib['id']) > 2:
					raw.write(token.find('lemma').text+' ')
			raw.write('\t{}\n'.format(sys.argv[2]))

if __name__ == "__main__":
	main(sys.argv[1:])