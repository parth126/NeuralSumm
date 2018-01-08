import math

class computeDF():

   int document_count 
   int sentence_count
   sf = {}
   df = {}

   def update_df(self, sentences):
      
      self.document_count += 1
      doc = set()
      for sentence in sentences:
         self.sentence_count += 1
         for term in set(sentence):
            doc.add(term)
            self.sf.get(term,0) += 1

      for term in doc:
         self.df.get(term,0) += 1
      return 0

   def get_idf(self):
      idf = {}
      for term in self.df:
         idf[term] = math.log(self.document_count / (1 + self.df[term]))
      return(idf)

   def get_isf(self):
      isf = {}
      for term in self.sf:
         isf[term] = math.log(self.sentence_count / (1 + self.sf[term]))
      return(isf)


def compute_tfidf(self, sentence, idf):
   tf = {}
   tf_idf = {}

   for term in sentence:
      tf.get(term,0) += 1
   for term in tf:
      tf_idf.get[term] = tf.get(term)*idf.get(term)
   return(tf_idf)







