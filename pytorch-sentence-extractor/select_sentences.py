import os
import re
import pickle
import nltk
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
from pprint import pprint
import io
import argparse
from utils import get_initial_datapath

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def clean_document(document):
    """Cleans document by removing unnecessary punctuation. It also removes
    any extra periods and merges acronyms to prevent the tokenizer from
    splitting a false sentence

    """
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = document.replace('-', '')
    document = document.replace('...', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove Ancronymns M.I.T. -> MIT
    # to help with sentence tokenizing
    document = merge_acronyms(document)

    # Remove extra whitespace
    document = ' '.join(document.split())
    return document

def remove_stop_words(document):
    """Returns document without stop words"""
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def similarity_score(t, s):
    """Returns a similarity score for a given sentence.

    similarity score = the total number of tokens in a sentence that exits
                        within the title / total words in title

    """
    t = remove_stop_words(t.lower())
    s = remove_stop_words(s.lower())
    t_tokens, s_tokens = t.split(), s.split()
    similar = [w for w in s_tokens if w in t_tokens]
    score = (len(similar) * 0.1 ) / len(t_tokens)
    return score

def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.',''))
    return s

def rank_sentences(doc, abs_sent, doc_matrix, feature_names, top_n=2):
    """Returns top_n sentences. Theses sentences are then used as summary
    of document.

    input
    ------------
    doc : a document as type str
    doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
    feature_names : a list of all features, the index is used to look up
                    tf-idf scores in the doc_matrix
    top_n : number of sentences to return

    """
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                  for sent in sentences]
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                 for sent in sentences]
    # Calculate Sentence Values
    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Similariy Score Weightings
    similarity_scores = [similarity_score(abs_sent, sent) for sent in sents]
    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    # Apply Position Weights
    ranked_sents = [sent*(i/len(sent_values))
                    for i, sent in enumerate(sent_values)]

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)

    return ranked_sents[:top_n]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select sentences matching')
    parser.add_argument('--legal', action="store_true",
                    help='Run System on Legal data')
    args = parser.parse_args()
    initial_data_path = get_initial_datapath(args)
    # Load corpus data used to train the TF-IDF Transformer
    data = pickle.load(open(initial_data_path + 'data.pkl', 'rb'))
    train_data = set(data)
    #print train_data
    # Fit and Transform the term frequencies into a vector
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(train_data)
    freq_term_matrix = count_vect.transform(train_data)
    feature_names = count_vect.get_feature_names()

    # Fit and Transform the TfidfTransformer
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    # Merge corpus data and new document data
    files = os.listdir(initial_data_path +  'processed_papers/')
    json_array = {}
    for file in files:
        json_internal_array = {}
        json_internal_array['document_id'] = file
    	json_internal_array['matched_sentences'] = {}
        abstract = ''
    	sections = ''
    	abs_sentences = {}
    	body_sentences = {}
        print(file)
        json_data = json.load(open(initial_data_path + 'processed_papers/'+file))
        if json_data["abstract_sentences"] is not None:
	    abs_sentences = json_data["abstract_sentences"]
        if json_data["body_sentences"] is not None:
	    body_sentences = json_data["body_sentences"]
        if (len(abs_sentences) == 0) or (len(body_sentences) == 0):
            json_array[file] = json_internal_array
            continue
        body_freq_term_matrix = count_vect.transform([ clean_document(body['sentence']) for body in body_sentences.values()])
        body_sentences_idx = list(body_sentences.keys())
        body_tfidf_matrix = tfidf.transform(body_freq_term_matrix)
    	abs_freq_term_matrix = count_vect.transform([ clean_document(abstract['sentence']) for abstract in abs_sentences.values()])
    	abs_sentences_idx = list(abs_sentences.keys())
    	abs_tfidf_matrix = tfidf.transform(abs_freq_term_matrix)
    	body_score = np.sum(body_tfidf_matrix, axis=1)/np.sum(body_tfidf_matrix>0, axis=1)
    	abs_score = np.sum(abs_tfidf_matrix, axis=1)/np.sum(abs_tfidf_matrix>0, axis=1)
    	kson_data_internal = {}

        #Dot product and find sentences
        resulting = abs_tfidf_matrix.dot(body_tfidf_matrix.T)
        i = 0
        for result in resulting:
	    #print "Abs_Sent" + abs_sentences[i] + " Body Sentence " + body_sentences[result.argmax()]
            json_internal_array['matched_sentences'][abs_sentences_idx[i]] = body_sentences_idx[result.argmax()]
            i = i + 1
        json_array[file] = json_internal_array

    	new_abs_id = 0
    	kson_data_internal["abstract_sentences"] = {}
    	for ids in abs_sentences_idx:
    	    kson_data_internal["abstract_sentences"][ids] = {}
    	    kson_data_internal["abstract_sentences"][ids]['sentence'] = abs_sentences[ids]['sentence']
    	    kson_data_internal["abstract_sentences"][ids]['section_heading'] = abs_sentences[ids]['section_heading']
    	    kson_data_internal["abstract_sentences"][ids]['tfidf'] = abs_score.item(new_abs_id)
    	    new_abs_id = new_abs_id + 1

    	new_body_id = 0
    	kson_data_internal["body_sentences"] = {}
    	for ids in body_sentences_idx:
    	    kson_data_internal["body_sentences"][ids] = {}
    	    kson_data_internal["body_sentences"][ids]['sentence'] = body_sentences[ids]['sentence']
    	    kson_data_internal["body_sentences"][ids]['section_heading'] = body_sentences[ids]['section_heading']
    	    kson_data_internal["body_sentences"][ids]['tfidf'] = body_score.item(new_body_id)
    	    new_body_id = new_body_id + 1
        with io.open(initial_data_path + 'tfidfprocessed_papers/' + file, 'w', encoding='utf8') as json_file:
            j = json.dumps(kson_data_internal, indent=4, sort_keys=True, ensure_ascii=False)
            json_file.write(unicode(j))
            json_file.close()

    with io.open(initial_data_path + 'sentences.json', 'w', encoding='utf8') as json_file:
        j = json.dumps(json_array, indent=4, sort_keys=True, ensure_ascii=False)
        json_file.write(unicode(j))
        json_file.close()
