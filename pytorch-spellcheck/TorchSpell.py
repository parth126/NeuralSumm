import torch.nn as nn
import torch
import torch.nn.init
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, ntoken, nhid, nlayers, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = nhid
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(ntoken, ntoken)
        self.embedding.weight.requires_grad=False
        self.encoder_i = getattr(nn, rnn_type)(ntoken, nhid, dropout=dropout, bidirectional=True)
        self.encoder_h = getattr(nn, rnn_type)(2*nhid, nhid, dropout=dropout, bidirectional=True)
        self.init_weights()

    def init_weights(self):
        ''' Initialise weights of embeddings '''
        torch.nn.init.eye(self.embedding.weight)
        ''' Initialise weights of encoder RNN '''
        torch.nn.init.xavier_normal(self.encoder_i.weight_ih_l0)
        torch.nn.init.xavier_normal(self.encoder_i.weight_hh_l0)
        torch.nn.init.xavier_normal(self.encoder_h.weight_ih_l0)
        torch.nn.init.xavier_normal(self.encoder_h.weight_hh_l0)        

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.encoder_i(output, hidden)
        for i in range(self.nlayers - 1):
            output, hidden = self.encoder_h(output, hidden)
        return output, hidden

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(2, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(2, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(1, bsz, self.hidden_size).zero_())


class DecoderRNN(nn.Module):
    def __init__(self, rnn_type, ntoken, nhid, nlayers, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = nhid
        self.rnn_type = rnn_type

        self.decoder_i = getattr(nn, rnn_type)(2*nhid, nhid, dropout=dropout)
        self.decoder_o = getattr(nn, rnn_type)(nhid, nhid, dropout=dropout)
        self.out = nn.Linear(nhid, ntoken)
        self.softmax = nn.LogSoftmax()
        self.init_weights()

    def init_weights(self):

        ''' Initialise weights of decoder RNN '''        
        torch.nn.init.xavier_normal(self.decoder_i.weight_ih_l0)
        torch.nn.init.xavier_normal(self.decoder_i.weight_hh_l0)
        torch.nn.init.xavier_normal(self.decoder_o.weight_ih_l0)
        torch.nn.init.xavier_normal(self.decoder_o.weight_hh_l0)

        ''' Initialise weights of linear layer '''        
        nn.init.xavier_normal(self.out.weight)
        #self.out.bias.data.fill_(0)


    def forward(self, input, hidden):
        output = input
        #for i in range(self.nlayers):
        output, hidden = self.decoder_i(output, hidden)
        output, hidden = self.decoder_o(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(1, bsz, self.hidden_size).zero_())
