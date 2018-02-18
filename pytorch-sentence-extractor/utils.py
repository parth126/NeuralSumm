# encoding: utf-8
import torch
import numpy as np
import pickle
import os
import sys
import extract_features
import model as model
import errno


def get_models_dir(args):
    return_directory = args.data + "/models/"
    if args.legal:
        return_directory =  args.data + "/models/legal/"
    if args.debug:
        print("Using model directory : " + return_directory)
    return return_directory

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
        if int_n > 0.05:
            return bcolors.OKGREEN
        elif int_n < 0.01:
            return bcolors.FAIL
        elif int_n > 0.03:
            return bcolors.OKBLUE
        else:
            return bcolors.WARNING

def init_embedding(embedding_size, ndictionary, embedding_weights_df):
    if not ((ndictionary) or (embedding_weights_df )):
        return
    temp_embedding_weights = []
    temp_embedding_weights_object = {}
    found_embedding_weights = 0
    notfound_embedding_weights = 0
    for _,tok in embedding_weights_df.iterrows():
        token = tok['word']
        embedding = tok['embedding']
        if ndictionary.feature2idx.has_key(token):
            temp_embedding_weights_object[ndictionary.feature2idx[token]] = embedding
    for i in range(len(ndictionary.feature2idx)):
        if temp_embedding_weights_object.has_key(i):
            #print("Embedding size", i, len(temp_embedding_weights_object[i]), embedding_size)
            assert len(temp_embedding_weights_object[i]) == embedding_size
            temp_embedding_weights.append(temp_embedding_weights_object[i])
            found_embedding_weights += 1
        else:
            #print("Not found embedding ", i, ndictionary.idx2feature[i])
            tensorinit = torch.FloatTensor(1, embedding_size)
            numpyarrayinit = torch.nn.init.xavier_normal(tensorinit).numpy()[0].tolist()
            temp_embedding_weights.append(numpyarrayinit)
            notfound_embedding_weights += 1
    print("Found Embedding weights for: ", found_embedding_weights, " Not found for : ", notfound_embedding_weights)
    temp_embedding_weights = np.array(temp_embedding_weights, dtype='f')
    #print(temp_embedding_weights.shape)
    assert temp_embedding_weights.shape == (ndictionary.__len__(), embedding_size)
    return temp_embedding_weights
    #self.embedding.weight.data.copy_(torch.from_numpy(temp_embedding_weights))

def variable_summaries(var, epoch, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    mean = torch.mean(var)
    writer.add_scalar('data' + name + 'mean', mean, epoch)
    stddev = torch.sqrt(torch.mean((var - mean)*(var - mean)))
    writer.add_scalar('data' + name + 'stddev', stddev, epoch)
    writer.add_scalar('data' + name + 'max', torch.max(var), epoch)
    writer.add_scalar('data' + name + 'min', torch.min(var), epoch)
    writer.add_histogram('data' + name + 'histogram', var, epoch)

def load_vectorization(args, DataFile):
    nparray = np.zeros(0)
    try:
        nparray = np.load(args.data + '/vectorization/' + DataFile + '.numpyarray.npy')
        vectorize = False
        print("Using exisiting numpy array")
    except:
        vectorize = True
        print("Couldnot load exisiting numpy for " + DataFile +". Will require vectorization")
        return nparray, vectorize
    return torch.from_numpy(nparray), vectorize

def save_vectorization(args, DataFile, nparray):
    try:
        os.mkdir(args.data + '/vectorization/')
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(' Vectorization Directory already exists. Existing models might be overwritten.')
        else:
            raise
    try:
        filename = args.data + '/vectorization/' + DataFile + '.numpyarray'
        np.save(filename, nparray)
        print("Saved numpy array to ", filename)
    except:
        print("Couldnot save numpy array!!")

def load_models(args, build=False):
    try:
        with open(get_models_dir(args) + 'Encoder.pt', 'rb') as f1:
            Encoder = torch.load(f1)
        with open(get_models_dir(args) + 'Classifier.pt', 'rb') as f2:
            Classifier = torch.load(f2)
            print("Using existing models")
    except IOError as e:
        print("Error: ", os.strerror(e.errno))
        if build:
            print("Could not load existing models. Building from scratch")
            Encoder = None
            Classifier = None
        else:
            print("Could not load existing models. Do the files exist?")
            sys.exit(-1)
    return Encoder, Classifier

def build_model_from_scratch(args, corpus, embed_df):
    print("Building the initial models")
    ntokens = len(corpus.dictionary)
    ntopic = args.ntopic
    iembedding_tensor = init_embedding(args.embed, corpus.dictionary, embed_df)
    Encoder = model.EncoderRNN(args.model, ntokens, args.embed, args.nhid, args.nlayers, args.dropout, iembedding_tensor)
    ''' Regular classifier without any attention
    Classifier = model.AttentionClassifier(ntopic, args.nhid, args.hhid, args.cembed,  args.max_len, args.dropout)
    '''
    Classifier = model.AttentionClassifier(ntopic, args.nhid, args.hhid, args.cembed,  args.max_len, args.dropout)

    if args.cuda:
        Encoder = Encoder.cuda()
        Classifier = Classifier.cuda()
    return Encoder, Classifier

def build_dictionary_from_scratch(args, train_df, valid_df, eval_df):
    print("Building the term Dictionary")
    corpus = extract_features.Corpus()
    corpus.add_to_dict(train_df, 'unigrams', 'sentence')
    corpus.add_to_dict(valid_df, 'unigrams', 'sentence')
    corpus.add_to_dict(eval_df, 'unigrams', 'sentence')

    corpus.clean_dict(args.min_count)
    with open(get_models_dir(args) + 'corpus_dictionary.pkl', 'wb') as output:
        pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)
    return corpus

def load_dictionary(args):
    try:
        with open(get_models_dir(args) + 'corpus_dictionary.pkl', 'rb') as input:
            corpus = pickle.load(input)
            print("Using existing corpus Dictionary")
    except:
        print("Could not load existing corpus Dictionary. Does the file exist?")
        sys.exit(-1)
    return corpus
