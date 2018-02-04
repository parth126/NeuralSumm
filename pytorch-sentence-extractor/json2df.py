import os
import json
import pandas as pd
import numpy as np

sentence_matching = './data/sentences.json'
data_folder = './data/papers'

with open(sentence_matching) as f:
    f1 = json.load(f)

sent_mapping_flattened = []
for i in f1:
    sentence_mapping = f1[i]['matched_sentences']
    doc_id = f1[i]['document_id']
    for j in sentence_mapping:
        sent_mapping_flattened.append([doc_id, j.strip('A'), sentence_mapping[j].strip('S')])

DF = pd.DataFrame(sent_mapping_flattened, columns = ['doc_id', 'abstract_sid', 'body_sid'])

abstract = []
body = []

for i in os.listdir(data_folder):
    with open(data_folder+'/'+i) as f:
        f1 = json.load(f)
        abstract_sentences = f1["abstract_sentences"]
        body_sentences = f1["body_sentences"]
        for a in abstract_sentences:
            abstract.append([i, a, abstract_sentences[a]])
        for b in body_sentences:
            body.append([i, b, body_sentences[b]])

DF1 = pd.DataFrame(abstract, columns = ['doc_id', 'abstract_sid', 'abstract_sentence'])
DF2 = pd.DataFrame(body, columns = ['doc_id', 'body_sid', 'body_sentence'])

DF4 = pd.merge(DF,DF1, on=['doc_id', 'abstract_sid'])
DF5 = pd.merge(DF4,DF2, how='right',  on=['doc_id', 'body_sid'])
DF5.abstract_sid.fillna(1000, inplace=True)

DF6 = pd.DataFrame(columns = ['doc_id','body_sid','sentence', 'is_in_abstract'])

DF6[['doc_id','body_sid','sentence']] = DF5[['doc_id', 'body_sid', 'body_sentence']]
DF6['is_in_abstract'][DF5.abstract_sid == 1000] = 0
DF6['is_in_abstract'][DF5.abstract_sid != 1000] = 1
print("Here")
DF6.to_pickle('./data/acl_data.pkl')

flattened = []
with open('./data/topic_documentwise') as f5:
    f6 = json.load(f5)
    
for k in f6:
    flattened.append([k, f6[k]])

DF7 = pd.DataFrame(flattened, columns = ['doc_id', 'context'])

DF8 = pd.merge(DF6,DF7, on=['doc_id'])

DF8.to_pickle('./data/acl_data_context.pkl')

DF9 = DF8.reindex(np.random.permutation(DF8.index))

