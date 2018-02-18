import os
import json
import pandas as pd
import numpy as np
import random
from nltk import word_tokenize as wt

typ = "small" #("small" or "all")

if(typ == 'small'):
    sentence_matching = './data/sentences.json.small'
    data_folder = './data/papers.small'
    context_file = './data/topic_documentwise.small'
else:
    sentence_matching = './data/sentences.json'
    data_folder = './data/papers'
    context_file = './data/topic_documentwise'
    
min_words = 5
max_words = 50

sort_by = "length"  #("length" or "docid")

# Load sentence mapping to a dataframe
with open(sentence_matching) as f:
    f1 = json.load(f)

sent_mapping_flattened = []
for i in f1:
    sentence_mapping = f1[i]['matched_sentences']
    doc_id = f1[i]['document_id']
    for j in sentence_mapping:
        sent_mapping_flattened.append([doc_id, j.strip('A'), sentence_mapping[j].strip('S')])

DF = pd.DataFrame(sent_mapping_flattened, columns = ['doc_id', 'abstract_sid', 'body_sid'])

print("one complete")
# Load body sentences and abstract sentences into a separate dataframe

abstract = []
body = []
senlen = []

for i in os.listdir(data_folder):
    with open(data_folder+'/'+i) as f:
        f1 = json.load(f)
        abstract_sentences = f1["abstract_sentences"]
        body_sentences = f1["body_sentences"]
        for a in abstract_sentences:
            abstract.append([i, a, abstract_sentences[a]])
        for b in body_sentences:
            sent = body_sentences[b]
            body.append([i, b, sent, len(wt(sent))])

DF1 = pd.DataFrame(abstract, columns = ['doc_id', 'abstract_sid', 'abstract_sentence'])
DF2 = pd.DataFrame(body, columns = ['doc_id', 'body_sid', 'body_sentence', 'senlen'])

print("two complete")
# Merge the three dataframes 

DF4 = pd.merge(DF,DF1, on=['doc_id', 'abstract_sid'])
DF5 = pd.merge(DF4,DF2, how='right',  on=['doc_id', 'body_sid'])
DF5.abstract_sid.fillna(1000, inplace=True)

print("three complete")
# create target labelled dataframe
DF6 = pd.DataFrame(columns = ['doc_id','body_sid','sentence', 'is_in_abstract'])

DF6[['doc_id','body_sid','sentence', 'senlen']] = DF5[['doc_id', 'body_sid', 'body_sentence', 'senlen']]
DF6['is_in_abstract'][DF5.abstract_sid == 1000] = 0
DF6['is_in_abstract'][DF5.abstract_sid != 1000] = 1

print("four complete")
# Read context data into a dataframe
flattened = []
with open(context_file) as f5:
    f6 = json.load(f5)

for k in f6:
    flattened.append([k, f6[k]])

DF7 = pd.DataFrame(flattened, columns = ['doc_id', 'context'])
print("five complete")
# Merge context information into the original dataframe
DF8 = pd.merge(DF6,DF7, on=['doc_id'])

if(typ == 'small'):
    DF8.to_pickle('./data/acl_data.pkl.small')
else:
    DF8.to_pickle('./data/acl_data.pkl')

print("six complete")
#Filter too small or too large sentences
DF9 = DF8[(DF8.senlen > min_words) & (DF8.senlen < max_words)]

ListOfDocs = list(set(DF9.doc_id))
random.shuffle(ListOfDocs)

if(typ == 'small'):
    TrainDocs = ListOfDocs
    ValidDocs = ListOfDocs
    EvalDocs = ListOfDocs
else:
    TrainDocs = ListOfDocs[0:23000]
    ValidDocs = ListOfDocs[23000:25000]
    EvalDocs = ListOfDocs[25000:]


DFTrainPos = DF9[(DF9.doc_id.isin(TrainDocs))  & (DF9.is_in_abstract == 1)]
DFValidPos = DF9[(DF9.doc_id.isin(ValidDocs)) & (DF9.is_in_abstract == 1)]
DFEvalPos = DF9[(DF9.doc_id.isin(EvalDocs)) & (DF9.is_in_abstract == 1)]

DFTrainNeg = DF9[(DF9.doc_id.isin(TrainDocs))  & (DF9.is_in_abstract == 0)].sample(frac = 0.1, replace=False)
DFValidNeg = DF9[(DF9.doc_id.isin(ValidDocs)) & (DF9.is_in_abstract == 0)]
DFEvalNeg = DF9[(DF9.doc_id.isin(EvalDocs)) & (DF9.is_in_abstract == 0)]

DFTrain = DFTrainPos.append(DFTrainNeg, ignore_index=True)
DFValid = DFValidPos.append(DFValidNeg, ignore_index=True)
DFEval = DFEvalPos.append(DFEvalNeg, ignore_index=True)

if(sort_by == "length"):
    DFTrain = DFTrain.sort_values(by=['senlen','is_in_abstract'], ascending=[1, 1])
    DFValid = DFValid.sort_values(by=['senlen','is_in_abstract'], ascending=[1, 1])
    DFEval = DFEval.sort_values(by=['senlen','is_in_abstract'], ascending=[1, 1])
else:
    DFTrain = DFTrain.sort_values(by=['doc_id','is_in_abstract'], ascending=[1, 1])
    DFValid = DFValid.sort_values(by=['doc_id','is_in_abstract'], ascending=[1, 1])
    DFEval = DFEval.sort_values(by=['doc_id','is_in_abstract'], ascending=[1, 1])


print("seven complete")


if(typ == 'small'):
    DFTrain.to_pickle('./data/train_data.pkl.small')
    DFValid.to_pickle('./data/valid_data.pkl.small')
    DFEval.to_pickle('./data/eval_data.pkl.small')
else:
    if(sort_by == "length"):
        DFTrain.to_pickle('./data/train_data.pkl')
        DFValid.to_pickle('./data/valid_data.pkl')
        DFEval.to_pickle('./data/eval_data.pkl')
    else:
        print("here")
        DFTrain.to_pickle('./data/train_docwise.pkl')
        DFValid.to_pickle('./data/valid_docwise.pkl')
        DFEval.to_pickle('./data/eval_docwise.pkl')
    
