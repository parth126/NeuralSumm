# -*- coding: utf-8 -*-
'''
Author: Parth Mehta 
Email: parth.mehta126@gmail.com 
'''

import os
import torch
import ast
from itertools import izip
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
from nltk import ngrams
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, rand
import time

class Featurize(object):
    def __init__(self):
        self.stoplist = set(stopwords.words('english'))
        self.stemmer = EnglishStemmer()

    def get_features(self, sentence, feature_type):    
 
        # TODO: Find a better name for argument 'sentence'. It is not actually a sentence but can be a list as well.
        # TODO: Add functionality for accepting list instead of a single feature_type 

        if(feature_type == 'unigrams'):
            features = self.get_unigram_features(sentence)
            return(features)
        if(feature_type == 'bigrams'):
            features = self.get_bigram_features(sentence)
            return(features)
        if(feature_type == 'topics'):
            features = self.get_topic_features(sentence)
        else:
            raise("Feature Not defined")

    # Add new get_feature functions here
    def get_unigram_features(self, sentence):
        sentence = ''.join([i if ord(i) < 128 else ' ' for i in sentence])
        unigrams = word_tokenize(sentence)
        #unigrams = word_tokenize(sentence.lower())
        #f = [self.stemmer.stem(u) for u in unigrams if u not in self.stoplist]
        return(unigrams)

    def get_bigram_features(self, sentence):
        unigrams = word_tokenize(sentence.lower())
        f = ngrams([u for u in unigrams if u not in self.stoplist], 2)
        f = [' '.join(bigrams) for bigrams in f]
        return(f)
        
    '''  # Redundant   
    def get_topic_features(self, list_of_tuples):    
        topic_list = ast.literal_eval(list_of_tuples)
        weights = [w for t, w in topic_list]
        topics = [t for t, w in topic_list]
        return(weights, topics)
    '''   

class Dictionary(object):
    def __init__(self):
        ''' ☕ : Padding '''
        ''' ǫ : E.O.S. '''
        self.feature2idx = {'☕': 0, 'ǫ': 1, 'UNK': 2}
        self.feature2count = {'☕': 1, 'ǫ': 1, 'UNK': 1}
        self.idx2feature = {0: '☕', 1: 'ǫ', 2: 'UNK'}
        self.count = 3

    def add_feature(self, feature_string):
        if feature_string not in self.feature2idx:
            self.idx2feature[self.count] = feature_string
            self.feature2idx[feature_string] = self.count
            self.feature2count[feature_string] = 1
            self.count += 1
        else:
            self.feature2count[feature_string] += 1

    def __len__(self):
        return len(self.idx2feature)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.featurize = Featurize()

    def add_to_dict(self, DF, type_of_features, field):
        for nline, row in DF.iterrows():
            sentence = row[field]

            """ Compute list of features from the sentence """
            features = self.featurize.get_features(sentence, type_of_features)
            features += ['ǫ']

            """Add features in the sentence to the dictionary"""
            for feature in features:
                self.dictionary.add_feature(feature)

    def vectorize(self, DF, type_of_features, max_len, field, min_count=3, add_noise=0, amount_of_noise=0, max_noise_in_caption=0):
        """ Vectorize the file content and pad the sequences to same length """

        nlines = len(DF)

        idx_vectors = torch.LongTensor(nlines, max_len)

        nline = 0
        for _,row in DF.iterrows():
            sentence = row[field]   

            features = self.featurize.get_features(sentence, type_of_features)

            nword = 0  
            for feature in features:
                if(self.dictionary.feature2count[feature] > min_count):
                   idx_vectors[nline,nword] = self.dictionary.feature2idx[feature] 
                   nword += 1
                   if(nword > max_len-2):
                      break
                else:
                   idx_vectors[nline,nword] = self.dictionary.feature2idx['UNK'] 
                   nword += 1
                   if(nword > max_len-2):
                      break


            idx_vectors[nline,nword] = self.dictionary.feature2idx['ǫ']
            nword += 1

            for c in range(nword, max_len):
                idx_vectors[nline,nword] = self.dictionary.feature2idx['☕']
                nword += 1

            nline += 1

        return idx_vectors
        
    def vectorize_list(self, DF, type_of_features, num_topics, field):
        """ Vectorize the topics to a fixed length """    
        nlines = len(DF)
        #idx_vectors = torch.LongTensor(nlines, num_topics)
        idx_weights = torch.FloatTensor(nlines, num_topics).zero_()
        nline = 0
        for _,row in DF.iterrows():
            sentence = ast.literal_eval(row[field])          
            
            #weights, features = self.featurize.get_features(sentence, type_of_features)
 
            for feature, weight in sentence:
                idx_weights[nline, feature] = weight
                
            nline += 1
        return(idx_weights)        
                     
    def vectorize_single_list(self, topic_list, num_topics):
        """ Vectorize the topics to a fixed length """    
        
        sentence = ast.literal_eval(topic_list)          
        idx_weights = torch.FloatTensor(1, num_topics).zero_()
 
        for feature, weight in sentence:
            idx_weights[0, feature] = weight
                
        return(idx_weights)           
        
    def vectorize_string(self, input_string, type_of_features, max_len, min_count=0):
        """ Vectorize the file content and pad the sequences to same length """

        idx_vectors = torch.LongTensor(1, max_len)
        #print(idx_vectors)
        features = self.featurize.get_features(input_string, type_of_features)

        nword = 0  
        for feature in features:
            if(self.dictionary.feature2count[feature] > min_count):
               idx_vectors[0,nword] = self.dictionary.feature2idx[feature] 
               nword += 1
               if(nword > max_len-2):
                  break

        idx_vectors[0,nword] = self.dictionary.feature2idx['ǫ']
        nword += 1

        for c in range(nword, max_len):
            idx_vectors[0,nword] = self.dictionary.feature2idx['☕']
            nword += 1
        #print(idx_vectors)
        return idx_vectors
            
