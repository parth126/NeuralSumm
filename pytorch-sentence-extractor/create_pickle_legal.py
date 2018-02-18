import os
import json
from pprint import pprint
import pickle
import nltk
import io
files = open('data/legal_data/file_ids.txt', 'r').readlines()
data = []

def handle_section(section_content, section_heading, output_data, pkl_data, tag, index = 1):
    pkl_data.extend(section_heading)
    pkl_data.extend(section_content)
    for sent in section_content:
        output_data[tag][index] = {}
        output_data[tag][index]['sentence'] = sent
        output_data[tag][index]['section_heading'] = section_heading
        index += 1
    return index

for file in files:
    file = file.strip("\n")
    output_data = {}
    output_data['document_id'] = file
    output_data['abstract_sentences'] = {}
    output_data['body_sentences'] = {}
    pprint(file)
    headnote = open('data/legal_data/Supreme_Court/Headnote/' + file + '.headnote').readlines()
    judgement = open('data/legal_data/Supreme_Court/Judgement/' + file + '.judgement').readlines()
    file_content = []
    if headnote is not None:
        handle_section(headnote, "", output_data, file_content, "abstract_sentences", 1)
    if judgement is not None:
        handle_section(judgement, "", output_data, file_content, "body_sentences", 1)
    data.append(" \n ".join(file_content))
    with io.open('data/legal_data/processed_papers/'+file, 'w', encoding='utf8') as json_file:
    	j = json.dumps(output_data, indent=4, sort_keys=True, ensure_ascii=False)
        json_file.write(unicode(j))
        json_file.close()
output = open('data/legal_data/data.pkl', 'wb')
pickle.dump(data, output)
output.close()
