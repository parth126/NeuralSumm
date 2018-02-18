import re
import random
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
import os
import json
from gensim import corpora
import logging
import time
from nltk.stem.snowball import EnglishStemmer
import argparse
from utils import get_initial_datapath

class MyCorpus(corpora.TextCorpus):
    def __init__(self, input=None, path = '../processed_papers'):
        self.path = path
	super(corpora.TextCorpus, self).__init__()
        self.input = input
        self.dictionary = Dictionary()
        self.metadata = False
        self.dictionary.add_documents(self.get_texts())

    def __len__(self):
	files = os.listdir(self.path)
	return len(files)

    def get_texts(self):
        files = os.listdir(self.path)
        counteR = 0
        json_data={}
        for fl in files:
            #print(counteR, ': ', fl)
            if(counteR%1000 == 0):
                print(counteR)
            counteR += 1
            text = ''
            json_data = json.load(open(self.path + '/'+fl))
            if json_data["title"] is not None:
            	text += (json_data["title"] + " ")
            for  val in json_data["abstract_sentences"].values():
		if val is not None:
		    text += val + " "
            for val in json_data['body_sentences'].values():
		if val is not None:
		    text += val + " "
	    yield ie_preprocess(text)

def ie_preprocess(document):
    document = re.sub('[^A-Za-z ]+', '', document)
    document = [stemmer.stem(w) for w in nltk.word_tokenize(document) if w.lower() not in stop]
    return document


def filter_corpus(corpus, threshold):
    corpus2 = [c for c in corpus]   # To skip prerprocessing again when filtering
    tfidf = models.TfidfModel(corpus2, id2word = corpus.dictionary)

    low_value = threshold
    filtered_corpus = []

    for c in corpus2:

        low_value_words = [] #reinitialize to be safe. You can skip this.
        low_value_words = [id for id, value in tfidf[c] if value < low_value]
        tfidf_ids = [id for id, value in tfidf[c]]
        bow_ids = [id for id, value in c]
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
        new_bow = [b for b in c if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        #print(len(c))
        #print(len(tfidf[c]))
        #print(len(low_value_words))
        #print(len(new_bow))

        # Assign to filtered corpus
        filtered_corpus.append(new_bow)

    return(filtered_corpus)


def run(args):

    stop = stopwords.words('english')
    add_stopwords = ['said', 'mln', 'billion', 'million', 'pct', 'would', 'inc', 'company', 'corp']
    stop += add_stopwords
    stemmer = EnglishStemmer()
    initial_data_path = get_initial_datapath(args)

    path = initial_data_path + 'processed_papers'
    """Import the Reuters Corpus which contains 10,788 news articles"""
    initial_data_path += 'lda/'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    stime = time.time()
    corpus = corpora.MmCorpus(initial_data_path + 'original_corpus.mm')
    corpus_filtered = corpora.MmCorpus(initial_data_path + 'filtered_corpus.mm')
    corpus.dictionary = corpora.dictionary.Dictionary.load(initial_data_path + 'original_corpus.dict')
    # Read LDA model
    lda = models.LdaModel.load(initial_data_path + 'ldamodel')
    for topic in lda.show_topics():
        print ("TOPIC: ", topic)
    files = os.listdir(path)
    counteR = 0
    topic_json={}
    json_data={}
    for fl in files:
    	#print(counteR, ': ', fl)
    	if(counteR%1000 == 0):
    	    print(counteR)
    	counteR += 1
    	text = ''
    	json_data = json.load(open(path + '/'+fl))
        if json_data.has_key('title'):
            if json_data["title"] is not None:
                text += (json_data["title"] + " ")
    	for  val in json_data["abstract_sentences"].values():
    	    if val['sentence'] is not None:
                text += val['sentence'] + " "
    	for val in json_data['body_sentences'].values():
    	    if val['sentence'] is not None:
                text += val['sentence'] + " "
    	idocdow = corpus.dictionary.doc2bow(ie_preprocess(text))
    	prediction = lda[idocdow]
        topic_json[fl] = str(prediction)
    outputfile = open(initial_data_path + 'topic_documentwise', 'wb')
    j = json.dumps(topic_json, indent=4, sort_keys=True)
    print >> outputfile,j
    outputfile.close()
    print(time.time() - stime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select sentences matching')
    parser.add_argument('--legal', action="store_true",
                                help='Run System on Legal data')
    parser.add_argument('--load_existing', action="store_true",
                                help='Run System on Legal data')
    parser.add_argument('--num_topics', type=int, default=500,
                                help='Number of topics')
    parser.add_argument('--passes', type=int, default=2,
                                help='Passes ')
    parser.add_argument('--chunksize', type=int, default=1000,
                                help='Chunk Size')
    args = parser.parse_args()

    stop = stopwords.words('english')
    add_stopwords = ['said', 'mln', 'billion', 'million', 'pct', 'would', 'inc', 'company', 'corp']
    stop += add_stopwords
    stemmer = EnglishStemmer()
    run(args)
