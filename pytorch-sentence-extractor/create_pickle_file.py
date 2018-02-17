import os
import json
from pprint import pprint
import pickle
import nltk
import io
files = os.listdir('data/papers')
data = []

def handle_section(section_content, section_heading, output_data, pkl_data, tag, index = 1):
    pkl_data.append(section_heading)
    pkl_data.append(section_content)
    sentence_tokenized = nltk.sent_tokenize(section_content)
    if not sentence_tokenized:
        return index
    newline_processed_sentences = []
    for sentence in sentence_tokenized:
        if (sentence.find("\n")):
            newline_processed_sentences.extend(sentence.splitlines())
        else:
            newline_processed_sentences.append(sentence)
    for sent in newline_processed_sentences:
        output_data[tag][index] = {}
        output_data[tag][index]['sentence'] = sent
        output_data[tag][index]['section_heading'] = section_heading
        index += 1
    return index

for file in files:
    output_data = {}
    output_data['document_id'] = file
    output_data['abstract_sentences'] = {}
    output_data['body_sentences'] = {}
    pprint(file)
    json_data = json.load(open('data/papers/'+file))
    file_content = []
    abstract_content = json_data["metadata"]["abstractText"]
    if abstract_content is not None:
        handle_section(abstract_content, "Abstract", output_data, file_content, "abstract_sentences", 1)
    sections = json_data["metadata"]["sections"]
    if sections is None:
        continue
    index = 1
    any_body_sentence = False
    for section in sections:
        section_content = ""
        section_heading = ""
        if section['heading'] is not None:
            section_heading = section['heading']
        else:
            continue
        if section['text'] is not None:
            section_content = section['text']
        else:
            continue
        index = handle_section(section_content, section_heading, output_data, file_content, "body_sentences", index)
        any_body_sentence = True
    if not any_body_sentence:
        continue
    output_data['title'] = json_data["metadata"]["title"]
    data.append(" \n ".join(file_content))
    with io.open('data/processed_papers/'+file, 'w', encoding='utf8') as json_file:
    	j = json.dumps(output_data, indent=4, sort_keys=True, ensure_ascii=False)
        json_file.write(unicode(j))
        json_file.close()
output = open('data/data.pkl', 'wb')
pickle.dump(data, output)
output.close()
