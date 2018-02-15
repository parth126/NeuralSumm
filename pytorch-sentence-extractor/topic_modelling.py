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


class MyCorpus(corpora.TextCorpus):
    def __init__(self, input=None, path = './processed_papers.small'):
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
            #print fl
            if(counteR%1000 == 0):
                print(counteR)
            counteR += 1
            text = ''
            json_data = json.load(open(self.path + '/'+fl))
            text += json_data["title"] if json_data["title"] is not None
            text += " "
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
 

def run():

    stop = stopwords.words('english')
    add_stopwords = ['said', 'mln', 'billion', 'million', 'pct', 'would', 'inc', 'company', 'corp']
    stop += add_stopwords
    stemmer = EnglishStemmer()

    path = './processed_papers'
    """Import the Reuters Corpus which contains 10,788 news articles"""

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    stime = time.time()
    corpus = MyCorpus()
    corpus_filtered = filter_corpus(corpus)

    #dictionary = corpus.dictionary()

    # Build LDA model
    lda = models.LdaMulticore(corpus_filtered, id2word=corpus.dictionary, num_topics=50, passes = 10, workers=5, chunksize=2000)
    for topic in lda.show_topics():
        print ("TOPIC: ", topic)
    lda.save('ldamodel')
    corpora.MmCorpus.save_corpus('original_corpus.mm', corpus_filtered)    
    corpora.MmCorpus.save_corpus('filtered_corpus.mm', corpus_filtered)
    print(time.time() - stime)

if __name__ == '__main__':
    run()
