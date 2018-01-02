# -*- coding: utf-8 -*-

import os
import torch
from itertools import izip
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, rand

class Dictionary(object):
    def __init__(self):
        self.char2idx = {'☕': 0, 'ǫ': 1}
        self.char2count = {'☕': 1, 'ǫ': 1}
        self.idx2char = {0: '☕', 1: 'ǫ'}
        self.count = 2

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char[self.count] = char
            self.char2idx[char] = self.count
            self.char2count[char] = 1
            self.count += 1
        else:
            self.char2count[char] += 1

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def add_to_dict(self, path):
        """Add characters in a text file to the dictionary"""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            nlines = 0
            for line in f:
                chars = list(line.decode('utf-8').strip('\n')) + ['ǫ']
                nlines += 1
                for char in chars:
                    self.dictionary.add_char(char)

    def vectorize(self, path, max_len, add_noise=0, amount_of_noise=0, max_noise_in_caption=0):
        """ Vectorize the file content and pad the sequences to same length """
        with open(path, 'r') as f:
            nlines = sum([1 for line in f])

        with open(path, 'r') as f:
            idx_vectors = torch.LongTensor(nlines, max_len)
            for nline,line in izip(range(nlines),f):
                nchar = 0
                chars = line.decode('utf-8').strip('\n') 
                if(add_noise):
                    flag = 0
                    added_noise = rand()
                    if added_noise < amount_of_noise:
                        for i in range(random_choice(range(1, max_noise_in_caption+1))):
                            noise_type = rand()
                            noise_position = random_randint(len(chars)-2)
                            random_char = self.dictionary.idx2char[random_choice(range(2, self.dictionary.count))]  # Used only for replace or insert noise
                            if noise_type < 0.25:    # Delete a character
                                chars = chars[:noise_position] + chars[noise_position + 1:]
                            elif noise_type < 0.5:   # Insert a character
                                chars = chars[:noise_position] + random_char + chars[noise_position:] 
                            elif noise_type < 0.75:  # Replace a character
                                chars = chars[:noise_position] + random_char + chars[noise_position + 1:]
                            else:
                                chars = chars[:noise_position] + chars[noise_position + 1] + chars[noise_position] + chars[noise_position + 2:]
                chars = list(chars) + ['ǫ'] 
                for char in chars:
                    idx_vectors[nline,nchar] = self.dictionary.char2idx[char]
                    nchar += 1
                    if(nchar > max_len-2):
                        idx_vectors[nline,nchar] = self.dictionary.char2idx['ǫ']
                        break
                for c in range(nchar, max_len):
                        idx_vectors[nline,nchar] = self.dictionary.char2idx['☕']
                        nchar += 1
        return idx_vectors
