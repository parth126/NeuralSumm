import math
from nltk.tokenize import sent_tokenize, word_tokenize

class computeDF():

   document_count = 0
   sentence_count = 0
   sf = {}
   df = {}

   def update_df(self, sentences):
      
      self.document_count += 1
      doc = set()
      for sentence in sentences:
         self.sentence_count += 1
         for term in set(sentence):
            doc.add(term)
            self.sf[term] = self.sf.get(term,0) + 1

      for term in doc:
         self.df[term] = self.df.get(term,0) + 1
      return 0

   def get_idf(self):
      idf = {}
      for term in self.df:
         idf[term] = math.log(self.document_count*1.0 / (1 + self.df[term]))
      return(idf)

   def get_isf(self):
      isf = {}
      for term in self.sf:
         isf[term] = math.log(self.sentence_count / (1 + self.sf[term]))
      return(isf)


def compute_tfidf(sentence, idf):
   tf = {}
   tf_idf = {}

   for term in sentence:
      tf[term] = tf.get(term,0) + 1
   for term in tf:
      tf_idf[term] = tf.get(term)*idf.get(term)
   return(tf_idf)


DF = computeDF()

with open('D30003') as f:
   p = f.read()
   sentence_list = []
   sentences = sent_tokenize(p)
   for s in sentences:
      words = word_tokenize(s)
      sentence_list.append(words)
   DF.update_df(sentence_list)

   idf = DF.get_idf()

   for s in sentence_list:
      tfIdf = compute_tfidf(s, idf)
      print(s)
      print(tfIdf)	



