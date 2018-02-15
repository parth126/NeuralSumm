import torch.nn as nn
import torch
import torch.nn.init
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, ntoken, nembedding, nhid, nlayers, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.embedding_size = nembedding
        self.lstm_hidden_size = nhid
        self.rnn_type = rnn_type
        self.dropout_p = dropout
        self.embedding = nn.Embedding(self.ntoken, self.embedding_size)
        self.embedding.weight.requires_grad=True
        self.encoder_i = getattr(nn, rnn_type)(self.embedding_size, self.lstm_hidden_size, dropout=self.dropout_p, bidirectional=False)
        self.encoder_h = getattr(nn, rnn_type)(self.lstm_hidden_size, self.lstm_hidden_size, dropout=self.dropout_p, bidirectional=False)

        self.dropout = nn.Dropout(self.dropout_p)
        self.init_weights()

    def init_weights(self):
        ''' Initialise weights of embeddings '''
        torch.nn.init.xavier_normal(self.embedding.weight)
        ''' Initialise weights of encoder RNN '''
        torch.nn.init.xavier_normal(self.encoder_i.weight_ih_l0)
        torch.nn.init.xavier_normal(self.encoder_i.weight_hh_l0)
        torch.nn.init.xavier_normal(self.encoder_h.weight_ih_l0)
        torch.nn.init.xavier_normal(self.encoder_h.weight_hh_l0)        

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded).unsqueeze(0)
        output, hidden = self.encoder_i(embedded, hidden)
        for i in range(self.nlayers - 1):
            output, hidden = self.encoder_h(output, hidden)

        return output, hidden

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(1, bsz, self.lstm_hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.lstm_hidden_size).zero_()))
        else:
            return Variable(weight.new(1, bsz, self.lstm_hidden_size).zero_())


class Classifier(nn.Module):

    def __init__(self, ntopic, nhid, hhid, cembed, dropout=0.5):
        super(Classifier, self).__init__()
        self.lstm_hidden_size = nhid
        self.context_embedding_size = cembed
        self.classifier_hidden_size = hhid
        self.sigmoid_activation = nn.Sigmoid()
        self.dropout_p = dropout
        self.ntopic = ntopic
        self.hidden_layer = nn.Linear(self.lstm_hidden_size + self.context_embedding_size, self.classifier_hidden_size)
        self.output_layer = nn.Linear(self.classifier_hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(self.ntopic, self.context_embedding_size)
        self.embedding.weight.requires_grad=True

        self.init_weights()

    def init_weights(self):
        ''' Initialise weights of embeddings '''
        torch.nn.init.xavier_normal(self.embedding.weight)
        ''' Initialise weights of hidden and output layer '''
        torch.nn.init.xavier_normal(self.hidden_layer.weight)
        torch.nn.init.xavier_normal(self.output_layer.weight)

    def forward(self, sentence, context, context_weights):

        embedded_context = self.embedding(context)
        
        embedded_context = self.dropout(embedded_context)
        
        context_embedding = torch.mm(context_weights ,embedded_context)
        
        merged_input = torch.cat((sentence, context_embedding),1)

       	classifier_hidden = self.hidden_layer(merged_input)
        classifier_hidden = self.dropout(classifier_hidden)

        classifier_output = self.output_layer(classifier_hidden)
        classifier_output = self.sigmoid_activation(classifier_output)

        return classifier_output
        
        
class AttentionClassifier(nn.Module):

    def __init__(self, ntopic, nhid, hhid, cembed, max_len, dropout=0.5):
        super(AttentionClassifier, self).__init__()
        self.lstm_hidden_size = nhid
        self.context_embedding_size = cembed
        self.classifier_hidden_size = hhid
        self.sigmoid_activation = nn.Sigmoid()
        self.dropout_p = dropout
        self.ntopic = ntopic
        self.max_len = max_len
        self.attn = nn.Linear(self.lstm_hidden_size + self.context_embedding_size, self.max_len)
        self.hidden_layer = nn.Linear(self.lstm_hidden_size + self.context_embedding_size, self.classifier_hidden_size)
        self.output_layer = nn.Linear(self.classifier_hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(self.ntopic, self.context_embedding_size)
        self.embedding.weight.requires_grad=True
        self.softmax = nn.Softmax(1)
        self.init_weights()

    def init_weights(self):
        ''' Initialise weights of embeddings '''
        torch.nn.init.xavier_normal(self.embedding.weight)
        ''' Initialise weights of hidden and output layer '''
        torch.nn.init.xavier_normal(self.hidden_layer.weight)
        torch.nn.init.xavier_normal(self.output_layer.weight)

    def forward(self, sentence_embedding, encoder_outputs, context, context_weights):

        embedded_context = self.embedding(context) 
        embedded_context = self.dropout(embedded_context)
        context_embedding = torch.mm(context_weights ,embedded_context)
        
        merged_input = torch.cat((sentence_embedding, context_embedding),1)
        
        ''' Attention Layer ''' 
        ''' Assign weights to each input timestamp, based on contet embedding and sentence embedding '''
        Attn = self.attn(merged_input)  
        attn_weights = self.softmax(Attn)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0).transpose(0,1), encoder_outputs.transpose(0,1))
        attn_applied = attn_applied.transpose(0,1)

        attn_combined = torch.cat((attn_applied[0], context_embedding), 1)
       	classifier_hidden = self.hidden_layer(attn_combined)
        classifier_hidden = self.dropout(classifier_hidden)

        classifier_output = self.output_layer(classifier_hidden)
        classifier_output = self.sigmoid_activation(classifier_output)
        
        return classifier_output, attn_weights       

