if [[ $1 -eq 'r' ]]; then
	mv ./$2 tweet.txt
	mv tweet.txt ../../stanford-corenlp
	cd ../../stanford-corenlp
	touch config.properties
	echo "ssplit.eolonly=true" >> config.properties
	java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,cleanxml -props config.properties -file tweet.txt -encoding "ascii"
	mv tweet.txt.xml ../compling/sent-tokenization/corpora
	cd ../compling/sent-tokenization/corpora
	python parseXML.py
	cd ..
	python main.py sarcastic
	mv ./$3 tweet.txt
	mv tweet.txt ../../stanford-corenlp
	cd ../../stanford-corenlp
	touch config.properties
	echo "ssplit.eolonly=true" >> config.properties
	java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,cleanxml -props config.properties -file tweet.txt -encoding "ascii"
	mv tweet.txt.xml ../compling/sent-tokenization/corpora
	cd ../compling/sent-tokenization/corpora
	python parseXML.py
	cd ..
	python main.py genuine
else
	echo "Place stanford-corenlp directory two directories above and rename it stanford-corenlp, then run this script with the -r parameter and the second parameter as the input file"
fi