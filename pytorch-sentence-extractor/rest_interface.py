# -*- coding: utf-8 -*-
from interfacetorch3 import PredictRestInterface 
from flask import Flask, jsonify, request
import torch
from torch.autograd import Variable
import sys
import pickle
import os
import json


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def return_color(int_nu):
        int_n = int_nu[0]
        if int_n > 0.022:
            return bcolors.OKGREEN
        elif int_n < 0.015:
            return bcolors.FAIL
        elif int_n > 0.015:
            return bcolors.OKBLUE
        else:
            return bcolors.WARNING
            
def Idx2word(indexes):
    restricted_chars = [ '☕' , 'ǫ' , u'\u201c' ]
    if indexes not in restricted_chars:
        return corpus.dictionary.idx2feature[indexes]
    else:
        return ''

def PrintRandomAttentionVisualization(input_variable, attention_weights):
    print(input_variable)
    input_data = input_variable.numpy()
    print(attention_weights)
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            print(bcolors.return_color(attention_weights[i, j].data.cpu().numpy()))
            print(Idx2word(input_data[i, j]), attention_weights[i, j].data.cpu().numpy()[0])
        print("")

sentenceExtractorApp = Flask(__name__)
        
@sentenceExtractorApp.route('/rate', methods=['POST'])
def check():
    content = request.json
    input_string = content['input'].encode('utf-8')
    context_string = content['context']
    
    if('max_len' in content):
        print("Here")
        max_len = content['max_len']
    else:
        max_len = int(40)
    if('isCuda' in content):
        isCuda = bool(content['isCuda'])
    else:
        isCuda = True
     
    input_variable = corpus.vectorize_string(input_string, 'unigrams', max_len, 0)
    
    if(isCuda):
        context_weights = Variable(corpus.vectorize_single_list(context_string, 500).transpose(0,1).cuda())
    else:
        context_weights = Variable(corpus.vectorize_single_list(context_string, 500))
           
    #print(context_weights)
    try:
       output, attention_weights = PredictRestInterface(input_variable, context_weights, Encoder, Classifier, max_len, isCuda)
       PrintRandomAttentionVisualization(input_variable, attention_weights)
       output = output[0]
       response = 200
    except Exception as e:
       print(e)
       response = 404
    return json.dumps({"output": str(output), "attention": str(attention_weights)}), response 
     
if __name__ == "__main__":
    try: 
        with open('./data/models/Encoder.pt', 'rb') as f1:
            Encoder = torch.load(f1)
        with open('./data/models/Classifier.pt', 'rb') as f2:
            Classifier = torch.load(f2)
        print("Using existing models")

    except IOError as e:
        print("Error: ", os.strerror(e.errno))
        print("Could not load existing models. Do the files exist?")
        sys.exit(-1)

    try:
        with open('./data/models/corpus_dictionary.pkl', 'rb') as input:
            corpus = pickle.load(input)
        print("Using existing Dictionary")
    except:
        print("Could not load existing Dictionary. Does the file exist?")
        sys.exit(-1)
   
    sentenceExtractorApp.run(host = "0.0.0.0" , port = 8092, debug=True)
