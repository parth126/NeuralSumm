# encoding: utf-8
from __future__ import print_function, unicode_literals
import argparse
import time
import math
import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import itertools
import numpy as np
import time
import math
import random
import os
import sys
import pickle
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

import prepare_data
import TorchSpell

parser = argparse.ArgumentParser(description='LSTM based Spell Checker')

parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--max_len', type=int, default=70,
                    help='Maximum sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--epoch', type=str,  default=500,
                    help='Number of Epochs to train')
parser.add_argument('--load-existing', action='store_true',
                    help='If existing models should be loaded')
parser.add_argument('--build_dict', action='store_true',
                    help='If character-index mapping needs to be built')
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
import time
start_time = time.time()

Train_Ifile = 'Cloudsight.wrong.train.small'
Train_Ofile = 'Cloudsight.right.train.small'
Test_Ifile = 'Cloudsight.incorrect.in.test'
Test_Ofile = 'Cloudsight.incorrect.out.test'
Valid_Ifile = 'Cloudsight.wrong.validate.small'
Valid_Ofile = 'Cloudsight.right.validate.small'

if(args.build_dict):
    print("Building the Dictionary")
    corpus = prepare_data.Corpus()
    corpus.add_to_dict(args.data + '/' + Train_Ifile)
    corpus.add_to_dict(args.data + '/' + Train_Ofile)
    corpus.add_to_dict(args.data + '/' + Valid_Ifile)
    corpus.add_to_dict(args.data + '/' + Valid_Ofile)
    with open('dictionary.pkl', 'wb') as output:
        pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)
else:
    print("Loading the Dictionary")
    with open('dict2.pkl', 'rb') as input:
        corpus = pickle.load(input)
#corpus.add_to_dict(args.data + '/' + Test_Ifile)
#corpus.add_to_dict(args.data + '/' + Test_Ofile)

print("Corpus Loaded")
train_ip = corpus.vectorize(args.data + '/' + Train_Ifile, args.max_len, 1, 0.5, 2)
train_op = corpus.vectorize(args.data + '/' + Train_Ofile, args.max_len)
valid_ip = corpus.vectorize(args.data + '/' + Valid_Ifile, args.max_len)
valid_op = corpus.vectorize(args.data + '/' + Valid_Ofile, args.max_len)
#test_ip = corpus.vectorize(args.data + '/' + Test_Ifile)
#test_op = corpus.vectorize(args.data + '/' + Test_Ofile)

print("Corpus Vectorized")
print("--- %s seconds ---" % (time.time() - start_time))


#-----------------------------------------------------------------------------#
# Build the model
#-----------------------------------------------------------------------------#

ntokens = len(corpus.dictionary)
print(ntokens, corpus.dictionary.count)
raw_input()
Encoder = TorchSpell.EncoderRNN(args.model, ntokens, args.nhid, args.nlayers, args.dropout)
Decoder = TorchSpell.DecoderRNN(args.model, ntokens, args.nhid, args.nlayers, args.dropout)
criterion = nn.NLLLoss()

if args.cuda:
    Encoder = Encoder.cuda()
    Decoder = Decoder.cuda()
print("Encoder: \n", Encoder)
print("Decoder: \n", Decoder)

#-----------------------------------------------------------------------------#
# Helper functions
#-----------------------------------------------------------------------------#

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, target, i, evaluation=False):
    Batch_len = min(args.batch_size, len(source) - i)
    source_batch = Variable(source[i-1:i+Batch_len-1], volatile=evaluation)
    target_batch = Variable(target[i-1:i+Batch_len-1])
    source_batch = source_batch.transpose(0,1).contiguous()
    target_batch = target_batch.transpose(0,1).contiguous()
    if args.cuda:
        source_batch = source_batch.cuda()
        target_batch = target_batch.cuda()
    return source_batch, target_batch

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def PrintRandomResults(encoder, decoder, n=10):
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
        print('Q', Q)
        print('A', Idx2sent(pair[1]))
        output, loss = Predict(Inp, Op, encoder, decoder)
        output = output.data.cpu().numpy().transpose()[0]
        print('P', Idx2sent(output))
        print("Correct?: ", Evaluate(Inp, Op, encoder, decoder), "Loss: ", loss)
        print('')

def Idx2sent(indexes):
    decoded_sent = ''.join(corpus.dictionary.idx2char[i].decode('utf-8') for i in indexes if i) 
    return decoded_sent


#-----------------------------------------------------------------------------#
# Training, Evaluation and Prediction functions
#-----------------------------------------------------------------------------#

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    loss = 0
    encoder_hidden = encoder.initHidden(input_variable.size(1))
    decoder_hidden = decoder.initHidden(input_variable.size(1))
    encoder_hidden = repackage_hidden(encoder_hidden)
    decoder_hidden = repackage_hidden(decoder_hidden)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
    encoder_last_output = encoder_output[-1,:,:].view(1,input_variable.size(1),-1)
    for d_step in range(args.max_len):
        decoder_output, decoder_hidden = decoder(encoder_last_output, decoder_hidden)
        loss += criterion(decoder_output, target_variable[d_step])
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / args.max_len

def Predict(input_variable, target_variable, encoder, decoder):
    loss = 0
    output = Variable(torch.LongTensor(input_variable.size(0),input_variable.size(1)).zero_())
    if(args.cuda):
        output = output.cuda()
    encoder_hidden = encoder.initHidden(input_variable.size(1))
    decoder_hidden = decoder.initHidden(input_variable.size(1))
    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
    encoder_last_output = encoder_output[-1,:,:].view(1,input_variable.size(1),-1)
    for d_step in range(args.max_len):
        decoder_output, decoder_hidden = decoder(encoder_last_output, decoder_hidden)
        loss += criterion(decoder_output, target_variable[d_step])
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0]
        #print("Topi: ", topi)
        output[d_step, :] = topi
    return(output, loss.data[0] / args.max_len)

def Evaluate(input_variable, target_variable, encoder, decoder):
    output, loss = Predict(input_variable, target_variable, encoder, decoder)
    output = output.transpose(0,1)
    target = input_variable.transpose(0,1)
    Correct = 0
    for i in range(output.size(0)):
        Correct += torch.equal(output.data[i,:], target.data[i,:])*1
    return(Correct, loss)

#-----------------------------------------------------------------------------#
# Training Iterator
#-----------------------------------------------------------------------------#

def trainIters(encoder, decoder, batch_size, print_every=100, plot_every=10, learning_rate=0.001):

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    train_ip = corpus.vectorize(args.data + '/' + Train_Ifile, args.max_len, 1, 0.2, 2)
    TrainData = torch.utils.data.TensorDataset(train_ip, train_op)
    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=args.batch_size, shuffle=True)
    ValData = torch.utils.data.TensorDataset(valid_ip, valid_op)
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=args.batch_size)

    for epoch in range(0, args.epoch):
        start = time.time()
        BatchN = 0.0
#-----------------------------------------------------------------------------#
        print("Evaluating the model")
        Eval_Correct = 0
        loss = 0.0
        BatchN = 0.0
        #for batch, iter in enumerate(range(1, valid_ip.size(0), args.batch_size)):
        #    input_variable, target_variable = get_batch(valid_ip, valid_op , iter)
        for input_v, target_v in ValDataLoader:
            input_variable = Variable(input_v).transpose(0,1).contiguous()
            target_variable = Variable(target_v).transpose(0,1).contiguous()
            if args.cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
            C, L = Evaluate(input_variable, target_variable, encoder, decoder)
            Eval_Correct += C
            loss += L
            BatchN += 1.0
    
        
        Eval_Accuracy = Eval_Correct*1.0/valid_ip.size(0)
        Eval_Loss = loss / BatchN
        Eval_Log1 = ('%s: %d, %s: %.4f, %s: %.4f' % ("Epoch", epoch, "Evaluation Accuracy", Eval_Accuracy, "Evaluation Loss", Eval_Loss))
        with open('./Evaluation.log', 'a') as f:
            f.write(Eval_Log1+'\n')    
        print(Eval_Log1)
        PrintRandomResults(encoder, decoder, 10)
#-----------------------------------------------------------------------------#

        #for batch, iter in enumerate(range(1, train_ip.size(0), args.batch_size)):
        #    BatchN = float(batch+1)
        for input_v, target_v in TrainDataLoader:
            BatchN += 1.0
            input_variable = Variable(input_v).transpose(0,1).contiguous()
            target_variable = Variable(target_v).transpose(0,1).contiguous()
            if args.cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()

            total_batches = (train_ip.size(0) - 1) / args.batch_size
         #   input_variable, target_variable = get_batch(train_ip, train_op , iter)
            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            if BatchN % print_every == 0:
                Log1 = ('%s: %d, %s: %d' % ("Epoch", epoch, "Batch", int(BatchN)))
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                Log2 = ('%s (%d%%) %s %.4f' % (timeSince(start, BatchN*1.0 / total_batches), BatchN / total_batches * 100, "Average Loss:", print_loss_avg))
                with open('./Training.log', 'a') as f:
                    f.write(Log1+'\n')
                    f.write(Log2+'\n')
                print(Log1)
                print(Log2)

            if BatchN % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        filename1 = ('%s%d%s' % ("Encoder.", epoch,".pt"))
        filename2 = ('%s%d%s' % ("Decoder.", epoch,".pt"))
        with open(filename1, 'wb') as f:
            torch.save(encoder, f)
        with open('Encoder.pt', 'wb') as f:
            torch.save(encoder, f)
        with open(filename2, 'wb') as f:
            torch.save(decoder, f)
        with open('Decoder.pt', 'wb') as f:
            torch.save(decoder, f)


        print("Evaluating the model")
        Eval_Correct = 0
        loss = 0.0
        BatchN = 0.0
        #for batch, iter in enumerate(range(1, valid_ip.size(0), args.batch_size)):
        #    input_variable, target_variable = get_batch(valid_ip, valid_op , iter)
        for input_v, target_v in ValDataLoader:
            input_variable = Variable(input_v).transpose(0,1).contiguous()
            target_variable = Variable(target_v).transpose(0,1).contiguous()
            if args.cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
            C, L = Evaluate(input_variable, target_variable, encoder, decoder)
            Eval_Correct += C
            loss += L
            BatchN += 1.0
    
        
        Eval_Accuracy = Eval_Correct*1.0/valid_ip.size(0)
        Eval_Loss = loss / BatchN
        Eval_Log1 = ('%s: %d, %s: %.4f, %s: %.4f' % ("Epoch", epoch, "Evaluation Accuracy", Eval_Accuracy, "Evaluation Loss", Eval_Loss))
        with open('./Evaluation.log', 'a') as f:
            f.write(Eval_Log1+'\n')    
        print(Eval_Log1)
        PrintRandomResults(encoder, decoder, 10)
        np.save('Losses.npy', plot_losses)
        #showPlot(plot_losses)

#-----------------------------------------------------------------------------#
# Training  	
#-----------------------------------------------------------------------------#
print("Preparing to Train the model")

if(args.load_existing):
    try: 
        with open('Encoder.pt', 'rb') as f1:
            Encoder = torch.load(f1)
        with open('Decoder.pt', 'rb') as f2:
            Decoder = torch.load(f2)
        print("Using existing models")

    except IOError as e:
        print("Error: ", os.strerror(e.errno))
        print("Could not load existing models. Do the files exist?")
        sys.exit(-1)

trainIters(Encoder, Decoder, args.batch_size)

