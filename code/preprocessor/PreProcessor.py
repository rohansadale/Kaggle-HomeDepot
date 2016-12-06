# -*- coding: utf-8 -*-
import re
import gensim
import time

from autocorrect import spell
from collections import namedtuple
from nltk.corpus import stopwords
from spell_corrector import str_stem
from TextCleaner import TextCleaner
import Stemmer

# TODO(akshay) - Try other Stemmer

class PreProcessor(object):

	TOKEN_PATTERN = "([\w][\w]*'?\w?)"
	LOCALE = "en"

	def __init__(self, data, description, attributes):
		self.data = data
		self.description = description
		self.attributes = attributes
		self.pattern = re.compile(PreProcessor.TOKEN_PATTERN)
		self.stemmer = Stemmer.Stemmer(PreProcessor.LOCALE)
		self.word2Vec_model_fname = '/Users/akshaykulkarni/Documents/study/ML/course/project/data/word2vec_dim500'
		self.doc2Vec_model_fname = '/Users/akshaykulkarni/Documents/study/ML/course/project/data/doc2vec_dim500'
		self.cachedStopWords = stopwords.words("english")

	def clean_it_(self, input_str):
		text_cleaner = TextCleaner(input_str)
		input_str = text_cleaner.transform()
		tokens = filter(lambda word: word not in self.cachedStopWords, self.pattern.findall(input_str))
		for i, entry in enumerate(tokens):
			tokens[i] = str(tokens[i].lower())
			tokens[i] = self.stemmer.stemWord(tokens[i])
			# TODO(akshay) - Think of more text cleaning approaches.
		return " ".join([entry for entry in tokens])

	def clean_(self):
		print "Cleaning product title .... "
		t1 = time.time()
		self.data['product_title'] = map(lambda x: self.clean_it_(x), self.data['product_title'])
		print "Time while cleaning product title is %d seconds \n" % int(time.time() - t1)

		print "Cleaning search term .... "
		t1 = time.time()
		self.data['search_term'] = map(lambda x:str_stem(x), self.data['search_term'])
		self.data['search_term'] = map(lambda x:self.clean_it_(x), self.data['search_term'])
		print "Time while cleaning search term is %d seconds \n" % int(time.time() - t1)

		print "Cleaning product description .... "
		t1 = time.time()
		self.description['product_description'] = map(lambda x: self.clean_it_(x), self.description.values())
		print "Time while cleaning product description is %d seconds \n" % int(time.time() - t1)

		print "Cleaning Attributes .... "
		t1 = time.time()
		self.attributes['name'] = self.attributes['name'].apply(lambda x: str(x).lower())
		self.attributes['value'] = self.attributes['value'].apply(lambda x: str(x).lower())
		print "Time while cleaning attributes is %d seconds \n" % int(time.time() - t1)

		print "Creating index for product_uid and product_title"
		self.product_title_map = {}
		for index, row in self.data.iterrows():
			self.product_title_map[row['product_uid']] = row['product_title']

	def construct_word2vec_model(self):
		print "Building Word2Vec model ..... "
		t1 = time.time()
		model = gensim.models.Word2Vec(self.description.values(), min_count = 5, size = 500, workers = 4, seed = 4)
		model.save(self.word2Vec_model_fname)
		print "Time while building word2Vec model is %d \n" % int(time.time() - t1)		

	def construct_doc2vec_model(self):
		print "Building doc2Vec model ..... "
		t1 = time.time()
		analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
		TaggedDocuments = []
		for i, sentence in enumerate(self.description.values()):
			try:
				TaggedDocuments.append(analyzedDocument(self.pattern.findall(sentence), [i]))
			except:
				pass
		model = gensim.models.Doc2Vec(TaggedDocuments, min_count = 5, size = 500, workers = 4, seed = 4)
		model.save(self.doc2Vec_model_fname)
		print "Time while building doc2Vec model is %d \n" % int(time.time() - t1)			

	def transform(self):
		print "cleaning data..."
		self.clean_()
		self.construct_word2vec_model()
		self.construct_doc2vec_model()