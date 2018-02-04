# encoding: utf-8
from __future__ import print_function, unicode_literals
import argparse
import time
import math
import torch
from torch import optim
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import itertools
import numpy as np
import time
import math
import random
import os
import sys
import pickle
import extract_features as ef
import pandas as pd
import errno
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import extract_features
import model

parser = argparse.ArgumentParser(description='LSTM based Spell Checker')

parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--hhid', type=int, default=1000,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--max_len', type=int, default=40,
                    help='Maximum sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--epoch', type=str,  default=500,
                    help='Number of Epochs to train')
parser.add_argument('--embed', type=float, default=100,
                    help='Character Embedding Size')
parser.add_argument('--load-existing', action='store_true',
                    help='If existing models should be loaded')
parser.add_argument('--build_dict', action='store_true',
                    help='If feature-index mapping needs to be built')
parser.add_argument('--mode', type=str, default='train',
                    help='Mode to use train/evaluate/predict')
parser.add_argument('--startepoch', type=int, default=0,
                    help='Start Epoch Number (for resuming training)')
parser.add_argument('--log_interval', type=int, default=50,
                    help='report interval')
parser.add_argument('--cembed', type=float, default=10,
                    help='context embedding size')
parser.add_argument('--ntopic', type=float, default=500,
                    help='maximum number of topics per document')                    

args = parser.parse_args()

# Set the random seed manually for reproducibility.

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: CUDA device available, running with --cuda might improve speed.")
    else:
        torch.cuda.manual_seed(args.seed)


#-----------------------------------------------------------------------------#
# Load and vectorize data
#-----------------------------------------------------------------------------#

print("Start")

Train_Data = 'train_context.pkl'
Valid_Data = 'valid_context.pkl'
Eval_Data = 'eval_context.pkl'


#-----------------------------------------------------------------------------#
# Helper functions
#-----------------------------------------------------------------------------#


class TensorContextDataset(torch.utils.data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, context_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        assert data_tensor.size(0) == context_tensor.size(0)
        self.data_tensor = data_tensor
        self.context_tensor = context_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.context_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)



"""Wraps hidden states in new Variables, to detach them from their history."""
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

''' Get the next batch from corpus '''
def get_batch(source, target, i, evaluation=False):   # i defines the batch number
    Batch_len = min(args.batch_size, len(source) - i)
    source_batch = Variable(source[i-1:i+Batch_len-1], volatile=evaluation)
    target_batch = Variable(target[i-1:i+Batch_len-1])
    source_batch = source_batch.transpose(0,1).contiguous()
    target_batch = target_batch.transpose(0,1).contiguous()
    if args.cuda:
        source_batch = source_batch.cuda()
        target_batch = target_batch.cuda()
    return source_batch, target_batch

''' Convert timestamp to minutes '''
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

''' Compute the spent and expected time for an epoch '''
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

''' Print random results for visualization
def PrintRandomResults(encoder, decoder, n=30):
    for i in range(n):
        pair = random.choice(zip(valid_ip, valid_op))
        if args.cuda:
            Inp = Variable(pair[0]).cuda()
            Op = Variable(pair[1]).cuda()
        else:
            Inp = Variable(pair[0])
            Op = Variable(pair[1])
        Inp = Inp.view(-1,1)
        Op = Op.view(-1,1)
        Q = Idx2sent(pair[0])

        output, loss, log_prob = Predict(Inp, Op, encoder, decoder)
        output = output.data.cpu().numpy().transpose()[0]
        print('Q', Q.encode('utf-8'))
        print('A', Idx2sent(pair[1]).encode('utf-8'))

        if(log_prob.data[:,0].cpu().numpy().mean() < args.threshold):
            P = Q.encode('utf-8')
        elif(round(loss,5) != round(log_prob.data[:,0].cpu().numpy().mean(),5)):
            P = Q.encode('utf-8')
        else:
            P = Idx2sent(output).encode('utf-8')

        print('P', P)
        print("Correct_2?: ", Idx2sent(output) == Idx2sent(pair[1]), "Log Likelihood: ", log_prob.data.cpu().numpy().transpose().mean(), "Loss: ", loss) 
        print('')
'''

''' Convert predicted indices to sentences
def Idx2sent(indexes):
    decoded_sent = ' '.join(corpus.dictionary.idx2char[i].decode('utf-8') for i in indexes if i)
    decoded_sent = decoded_sent.strip().strip('☕').strip('ǫ').strip() 
    return decoded_sent
'''

#-----------------------------------------------------------------------------#
# Training, Evaluation and Prediction functions
#-----------------------------------------------------------------------------#

def train(input_variable, target_variable, context_variable, encoder, encoder_optimizer, classifier, classifier_optimizer, criterion):

    ''' Initialization '''
    loss = 0
    encoder_hidden = encoder.initHidden(input_variable.size(1))
    encoder_outputs = Variable(torch.zeros(args.max_len, input_variable.size(1), encoder.lstm_hidden_size))
    encoder_outputs = encoder_outputs.cuda() if args.cuda else encoder_outputs
    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
 
    ''' Encoder Forward Pass '''
    for i_step in range(args.max_len):
        encoder_output, encoder_hidden = encoder(input_variable[i_step], encoder_hidden)
        encoder_outputs[i_step] = encoder_output[0] 
    #print("CC2: ", context_variable)
    #print("II2: ", encoder_output[0])
    ''' Regular classifier without any attention
    final_output = classifier(encoder_output[0], context_variable.transpose(0,1))
    ''' 
    final_output, attention_weights = classifier(encoder_output[0], encoder_outputs, context_variable.transpose(0,1))

    #criterion.weight = 0.02*torch.cuda.FloatTensor(target_variable.size())
    #criterion.weight[(target_variable.data == 1)] = 0.98

    loss += criterion(final_output, target_variable)
    
    ''' Back propogation '''
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)

    encoder_optimizer.step()
    classifier_optimizer.step()
    return loss.data[0]

def Evaluate(input_variable, target_variable, context_variable, encoder, classifier):
    output, loss = Predict(input_variable, target_variable, context_variable, encoder, classifier)
    output[output > 0.5] = 1.0
    output[output <= 0.5] = 0.0
    output = output.transpose(0,1)
    
    Correct = 0
    for i in range(output.size(1)):
        out = output.data[:,i]
        #print("A: ", out, target_variable.data[i,:])
        Correct += torch.equal(out, target_variable.data[:,i])*1
        #print("C:", Correct)
    return(Correct, loss)


def Predict(input_variable, target_variable, context_variable, encoder, classifier):

    ''' Initialization '''
    loss = 0
    encoder_hidden = encoder.initHidden(input_variable.size(1))
    encoder_outputs = Variable(torch.zeros(args.max_len, input_variable.size(1), encoder.lstm_hidden_size))
    encoder_outputs = encoder_outputs.cuda() if args.cuda else encoder_outputs

    ''' Encoder Forward Pass '''
    for i_step in range(args.max_len):
        encoder_output, encoder_hidden = encoder(input_variable[i_step], encoder_hidden)
        encoder_outputs[i_step] = encoder_output[0] 
    ''' Classifier forward pass '''
    
    ''' Regular classifier without any attention
    final_output = classifier(encoder_output[0], context_variable.transpose(0,1))
    ''' 
    final_output, attention_weights = classifier(encoder_output[0], encoder_outputs, context_variable.transpose(0,1))
    #print(attention_weights)
    plt.matshow(attention_weights.data.cpu().numpy())

    loss += criterion(final_output, target_variable)
 
    return(final_output, loss.data[0])


#-----------------------------------------------------------------------------#
# Training Iterator
#-----------------------------------------------------------------------------#

def trainIters(encoder, classifier, batch_size, print_every=100, learning_rate=0.0001):

    train_loss_total = 0  # Reset every print_every

    ''' define optimizer and loss function '''
    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    ''' Load the training and testing data '''

    TrainData = TensorContextDataset(train_ip, train_op, train_context_weights)
    #TrainData = torch.utils.data.TensorDataset(train_ip, train_op)
    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=args.batch_size, shuffle=True)
    ValData = TensorContextDataset(valid_ip, valid_op, valid_context_weights)
    #ValData = torch.utils.data.TensorDataset(valid_ip, valid_op)
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=args.batch_size)

    for epoch in range(args.startepoch, args.epoch):
        start = time.time()
        BatchN = 0.0

        ''' Iterate over all batches for an epoch '''
        #for input_v, target_v, context_v in TrainDataLoader:
        for input_v, target_v, context_v in TrainDataLoader:
            BatchN += 1.0
            input_variable = Variable(input_v).transpose(0,1).contiguous()
            target_variable = Variable(target_v).transpose(0,1).contiguous()
            context_variable = Variable(context_v).transpose(0,1).contiguous()
            #context_variable = Variable(input_v).transpose(0,1).contiguous()

            if args.cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
                context_variable = context_variable.cuda()
                
            #print("II: ", input_variable)
            #print("TT: ", target_variable)
            #print("CC: ", context_variable)

            total_batches = (train_ip.size(0) - 1) / args.batch_size

            loss = train(input_variable, target_variable, context_variable, encoder, encoder_optimizer, classifier, classifier_optimizer, criterion)
            train_loss_total += loss
            if BatchN % print_every == 0:
                Log1 = ('%s: %d, %s: %d' % ("Epoch", epoch, "Batch", int(BatchN)))
                train_loss_avg = train_loss_total / print_every
                train_loss_total = 0
                Log2 = ('%s (%d%%) %s %.4f' % (timeSince(start, BatchN*1.0 / total_batches), BatchN / total_batches * 100, "Average Loss:", train_loss_avg))
                with open('./Training.log', 'a') as f:
                    f.write(Log1+'\n')
                    f.write(Log2+'\n')
                print(Log1)
                print(Log2)

        ''' Save the loss info batchwise, for every epoch '''
        filename1 = ('%s%d%s' % ("Encoder.", epoch,".pt"))
        filename2 = ('%s%d%s' % ("Classifier.", epoch,".pt"))
        with open('./data/models/'+filename1, 'wb') as f:     # Saving the trained models for every epoch
            torch.save(encoder, f)
        with open('./data/models/Encoder.pt', 'wb') as f:  # Latest copy to be used in next iteration or when resuming training
            torch.save(encoder, f)  
        with open('./data/models/'+filename2, 'wb') as f:     # Saving the trained models for every epoch
            torch.save(classifier, f)
        with open('./data/models/Classifier.pt', 'wb') as f:  # Latest copy to be used in next iteration or when resuming training
            torch.save(classifier, f)


        ''' Evaluate on the validation set after each epoch '''
        print("Evaluating the model")
        Eval_Correct = 0
        loss = 0.0
        BatchN = 0.0

        #for input_v, target_v, context_v in ValDataLoader:
        for input_v, target_v, context_v in ValDataLoader:
            input_variable = Variable(input_v).transpose(0,1).contiguous()
            target_variable = Variable(target_v).transpose(0,1).contiguous()
            context_variable = Variable(context_v).transpose(0,1).contiguous()
            if args.cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
                context_variable = context_variable.cuda()

            C, L = Evaluate(input_variable, target_variable, context_variable, encoder, classifier)
            Eval_Correct += C
            loss += L
            BatchN += 1.0
    
        Eval_Accuracy = Eval_Correct*1.0/valid_ip.size(0)
        Eval_Loss = loss / BatchN
        Eval_Log = ('%s: %d, %s: %.4f, %s: %.4f' % ("Epoch", epoch, "Evaluation Accuracy", Eval_Accuracy, "Evaluation Loss", Eval_Loss))
        with open('./data/Evaluation.log', 'a') as f:
            f.write(Eval_Log+'\n')    
        print(Eval_Log)


#-----------------------------------------------------------------------------#
# Main interface  	
#-----------------------------------------------------------------------------#

if __name__== "__main__":

    try:
        os.mkdir('./data/models')
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists. Existing models might be overwritten.')
        else:
            raise

    if(args.mode =='train'):

        print("Preparing to Train the model")

        ''' Load and vectorize data '''

        print("Start")
        train_df = pd.read_pickle(args.data + '/' + Train_Data)
        valid_df = pd.read_pickle(args.data + '/' + Valid_Data)
        eval_df = pd.read_pickle(args.data + '/' + Eval_Data)

        if(args.load_existing):

            try: 
                with open('./data/models/Encoder.pt', 'rb') as f1:
                    Encoder = torch.load(f1)
                with open('./data/models/Decoder.pt', 'rb') as f2:
                    Classifier = torch.load(f2)
                print("Using existing models")

            except IOError as e:
                print("Error: ", os.strerror(e.errno))
                print("Could not load existing models. Building from scratch")
                print("Building the initial models")
                ntokens = len(corpus.dictionary)
                ntopic = args.ntopic
                Encoder = model.EncoderRNN(args.model, ntokens, args.embed, args.nhid, args.nlayers, args.dropout)
                ''' Regular classifier without any attention
                Classifier = model.AttentionClassifier(ntopic, args.nhid, args.hhid, args.cembed,  args.max_len, args.dropout)
                '''
                Classifier = model.AttentionClassifier(ntopic, args.nhid, args.hhid, args.cembed,  args.max_len, args.dropout)

                if args.cuda:
                   Encoder = Encoder.cuda()
                   Classifier = Classifier.cuda()

            try:
                with open('./data/models/corpus_dictionary.pkl', 'rb') as input:
                    corpus = pickle.load(input)
                print("Using existing corpus Dictionary")
            except:
                print("Could not load existing corpus Dictionary. Does the file exist?")
                sys.exit(-1)
            '''    
            try:
                with open('./data/models/context_dictionary.pkl', 'rb') as input:
                    context = pickle.load(input)
                print("Using existing context Dictionary")
            except:
                print("Could not load existing context Dictionary. Does the file exist?")
                sys.exit(-1)
            '''    
                
        else:

            print("Building the term Dictionary")
            corpus = extract_features.Corpus()
            corpus.add_to_dict(train_df, 'unigrams', 'sentence')
            corpus.add_to_dict(valid_df, 'unigrams', 'sentence')
            corpus.add_to_dict(eval_df, 'unigrams', 'sentence')
            
            with open('./data/models/corpus_dictionary.pkl', 'wb') as output:
                pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)

            ''' Redundant
            print("Building the context Dictionary")
            context = extract_features.Corpus()
            context.add_to_dict(train_df, 'unigrams', 'sentence')
            context.add_to_dict(valid_df, 'unigrams', 'sentence')
            context.add_to_dict(eval_df, 'unigrams', 'sentence')
            
            with open('./data/models/context_dictionary.pkl', 'wb') as output:
                pickle.dump(context, output, pickle.HIGHEST_PROTOCOL)
            '''
            
            print("Building the initial models")
            ntokens = len(corpus.dictionary)
            ntopic = args.ntopic
            #ntopic = len(context.dictionary)
            #print("N: ", ntopic)
            #raw_input()
            Encoder = model.EncoderRNN(args.model, ntokens, args.embed, args.nhid, args.nlayers, args.dropout)
            ''' Regular classifier without any attention
            Classifier = model.AttentionClassifier(ntopic, args.nhid, args.hhid, args.cembed,  args.max_len, args.dropout)
            '''
            Classifier = model.AttentionClassifier(ntopic, args.nhid, args.hhid, args.cembed,  args.max_len, args.dropout)

            if args.cuda:
                Encoder = Encoder.cuda()
                Classifier = Classifier.cuda()

        print("Dictionary and model built. Vectorizing the corpus now...")
         
        train_ip = corpus.vectorize(train_df, 'unigrams', args.max_len, 'sentence')
        train_op = torch.FloatTensor(np.expand_dims(train_df.is_in_abstract.as_matrix(), 1))
        train_context_weights = corpus.vectorize_list(train_df, 'topics', args.ntopic, 'context')

        valid_ip = corpus.vectorize(valid_df, 'unigrams', args.max_len, 'sentence')
        valid_op = torch.FloatTensor(np.expand_dims(valid_df.is_in_abstract.as_matrix(),1))
        valid_context_weights = corpus.vectorize_list(valid_df, 'topics', args.ntopic, 'context')

        print("Corpus and Context Vectorized. Starting Training...")

        criterion = nn.BCELoss()
        trainIters(Encoder, Classifier, args.batch_size, args.log_interval)

    elif(args.mode == 'evaluate'):
        print("Preparing to evaluate the model")

        ''' Load and vectorize data '''

        print("Start")

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
            
        '''    
        try:
            with open('./data/models/context_dictionary.pkl', 'rb') as input:
                context = pickle.load(input)
            print("Using existing context Dictionary")
        except:
            print("Could not load existing context Dictionary. Does the file exist?")
            sys.exit(-1)
        '''    

        print("Dictionary and model loaded. Vectorizing the corpus now...")

        eval_ip = corpus.vectorize(eval_df, 'unigrams', args.max_len, 'sentence')
        eval_op = torch.FloatTensor(np.expand_dims(eval_df.is_in_abstract.as_matrix(),1))
        eval_context_weights = context.vectorize_list(eval_df, 'topics', args.ntopic, 'context')

        print("Corpus and context Vectorized. Starting Evaluation...")

        criterion = nn.BCELoss()
        evalIters(Encoder, Classifier, args.batch_size, args.log_interval)

    elif(args.mode == 'predict'):
        print("Preparing to predict")

        ''' Load and vectorize data '''

        print("Start")

        try: 
            with open('./data/models/Encoder.pt', 'rb') as f1:
                Encoder = torch.load(f1)
            with open('./data/models/Decoder.pt', 'rb') as f2:
                Decoder = torch.load(f2)
            print("Using existing models")
        except IOError as e:
            print("Error: ", os.strerror(e.errno))
            print("Could not load existing models. Do the files exist?")
            sys.exit(-1)

        try:
            with open('./data/models/corpus_dictionary.pkl', 'rb') as input:
                corpus = pickle.load(input)
            print("Using existing corpus Dictionary")
        except:
            print("Could not load existing corpus Dictionary. Does the file exist?")
            sys.exit(-1)

        '''       
        try:
            with open('./data/models/context_dictionary.pkl', 'rb') as input:
                context = pickle.load(input)
            print("Using existing context Dictionary")
        except:
            print("Could not load existing context Dictionary. Does the file exist?")
            sys.exit(-1)
        '''

        print("Dictionary and model loaded. Vectorizing the corpus now...")

        predict_ip = corpus.vectorize(args.data + '/' + Prediction_Ifile, args.max_len, 'sentence')
        predict_op = corpus.vectorize(args.data + '/' + Prediction_Ofile, args.max_len)
        predict_context = context.vectorize(args.data + '/' + Prediction_Ifile, args.max_len, 'topic')
        
        print("Corpus Vectorized. Starting Prediction...")

        criterion = nn.NLLLoss()
        predictIters(Encoder, Decoder, args.batch_size, args.log_interval)
        
       
# TODO: include Predictiters and evaliters
# TODO: modify predict condition in main
