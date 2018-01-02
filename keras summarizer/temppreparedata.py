#!/home/parth/anaconda2/bin/python

import sys, getopt
import numpy as np
import string

Threshold = 2

def GetWords(l):
## Code for preprocessing sentences and generating word tokens
	return()

def GenVocab():
## Code for generating Vocabulary files with lookup by terms and term ids
	return()

def EncodeInput():
## Code for generating an encoded file ready to be used by the CNN module
	return()

def main(argv):
	inputfile = ''
	outputfile = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
		print opts
		print args
	except getopt.GetoptError:
		print 'preparedata.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'preparedata.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--inputfile"):
			inputfile = arg
		elif opt in ("-o", "--outputfile"):
			outputfile = arg

## Create Vocabulary with Lookup by terms and lookup by term ids
	TTable = string.maketrans(string.punctuation, " "*len(string.punctuation))
	Lookup = {}
	TermFreq = {}	
	UniqTerms = 4
	Input = open(inputfile, 'r')
	for line in Input:
		words = line.translate(TTable).split()
		for w in words:
			if not Lookup.has_key(w):
				Lookup[w] = UniqTerms
				UniqTerms += 1
				TermFreq[w] = 1
			else:
				TermFreq[w] += 1
	Input.close()

#	InverseLookup = {id: w for w, id in Lookup.items()}

## Lookup Embeddings
	W2V = {}
	f = open('/home/parth/Documents/Data/GoogleNews-vectors-negative300.txt')
	for line in f:
		values = line.split()
		Word = values[0]
		Coefs = np.asarray(values[1:], dtype='float32')
		W2V[Word] = Coefs
	f.close()
	print('Found %s word vectors.' % len(W2V))

## Save Relevant Embeddings

	LookupShort = {}
	TermFreqShort = {}

	Embeddings = {}
	Embeddings['<UNK>'] = 0.03*np.random.randn(300)
	Embeddings['<GO>'] = 0.03*np.random.randn(300)
	Embeddings['<EOS>'] = 0.03*np.random.randn(300)
	Embeddings['<PAD>'] = np.zeros(300)

	LookupShort['<PAD>'] = 0
	LookupShort['<UNK>'] = 1
	LookupShort['<GO>'] = 2
	LookupShort['<EOS>'] = 3

	TermFreqShort['<UNK>'] = 1000000
	TermFreqShort['<GO>'] = 1000000
	TermFreqShort['<EOS>'] = 1000000
	TermFreqShort['<PAD>'] = 1000000

	UniqTerms = 4

	for w in Lookup:
		if(TermFreq[w] > Threshold):  ## Ignoring low freq terms reduces size of Embeddings File
			LookupShort[w] = UniqTerms
			TermFreqShort[w] = TermFreq[w]
			UniqTerms += 1
			if(W2V.has_key(w)):
				Embeddings[w] = W2V[w]
			elif(W2V.has_key(w.lower())):
				Embeddings[w] = W2V[w.lower()]	
			else:
				Embeddings[w] = 0.03*np.random.randn(300)
	
	Lookup = LookupShort
	TermFreq = TermFreqShort
	InverseLookup = {id: w for w, id in Lookup.items()}

	Em = np.zeros((len(Embeddings),300))	
	for w in Embeddings:
		Em[Lookup[w],:] = Embeddings[w]

	np.save('Embeddings', np.array(Em))
	np.save('EmbeddingsWithNames', Embeddings)
	np.save('Lookup', Lookup)
	np.save('TermFreq', TermFreq)
	np.save('Inverselookup', InverseLookup)

	DocEnc = []
	Input = open('/home/parth/Documents/Data/neuralsum/cnn/USE/All.title', 'r')
	for line in Input:
		LineEnc = []
		words = line.translate(None, string.punctuation).split()
		for w in words:
			if not Lookup.has_key(w):
				LineEnc.append(Lookup.get('<UNK>'))
			else:
				LineEnc.append(Lookup.get(w))
		LineEnc.append(Lookup.get('<EOS>'))
		for i in range(len(LineEnc), 20):
			LineEnc.append(Lookup.get('<PAD>'))
		DocEnc.append(np.array(LineEnc))
		#print LineEnc
	Input.close()
	Output = open('/home/parth/Documents/Data/neuralsum/cnn/USE/TitleEnc.npy', 'w')
	np.save(Output, np.array(DocEnc))

	DocEnc = []
	Input = open('/home/parth/Documents/Data/neuralsum/cnn/USE/All.top1', 'r')
	for line in Input:
		LineEnc = []
		words = line.translate(None, string.punctuation).split()
		for w in words:
			if len(LineEnc) < 79:
				if not Lookup.has_key(w):
					LineEnc.append(Lookup.get('<UNK>'))
				else:
					LineEnc.append(Lookup.get(w))
		LineEnc.append(Lookup.get('<GO>'))
		for i in range(len(LineEnc), 80):
			LineEnc.append(Lookup.get('<PAD>'))
		DocEnc.append(np.array(LineEnc))
		#print LineEnc
	Input.close()
	Output = open('/home/parth/Documents/Data/neuralsum/cnn/USE/TopEnc.npy', 'w')
	np.save(Output, np.array(DocEnc))
	#print Lookup
	#print np.array(DocEnc)


if __name__ == "__main__":
	main(sys.argv[1:])
