#!/home/parth/anaconda2/bin/python

import sys, getopt
import numpy as np
import string

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

	Lookup = {}
	TermFreq = {}	
	UniqTerms = 1
	Input = open(inputfile, 'r')
	for line in Input:
		words = line.translate(None, string.punctuation).lower().split()
		for w in words:
			if not Lookup.has_key(w):
				Lookup[w] = UniqTerms
				UniqTerms += 1
				TermFreq[w] = 1
			else:
				TermFreq[w] += 1
	Input.close()

	Lookup['<UNK>'] = UniqTerms
	Lookup['<GO>'] = UniqTerms + 1
	Lookup['<EOS>'] = UniqTerms + 2
	Lookup['<PAD>'] = 0
	InverseLookup = {id: w for w, id in Lookup.items()}

	DocEnc = []
	Input = open(inputfile, 'r')
	for line in Input:
		LineEnc = []
		words = line.translate(None, string.punctuation).lower().split()
		for w in words:
			if not Lookup.has_key(w):
				LineEnc.append(Lookup.get('<UNK>'))
			else:
				LineEnc.append(Lookup.get(w))
		LineEnc.append(Lookup.get('<GO>'))
		for i in range(len(LineEnc), 120):
			LineEnc.append(Lookup.get('<PAD>'))
		DocEnc.append(np.array(LineEnc))
		#print LineEnc
	Input.close()
	Output = open(outputfile, 'w')
	np.save(Output, np.array(DocEnc))
	#print Lookup
	#print np.array(DocEnc)

if __name__ == "__main__":
	main(sys.argv[1:])
