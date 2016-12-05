# -*- coding: utf-8 -*-
import gensim
import sys
import time
sys.path.append("..")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.dist_util import compute_jaccard, compute_edit_distance, compute_first_last_intersect, compute_cosine_similarity
from utils.dist_util import compute_longest_match, compute_rmse, compute_match_attr_ratio, compute_coccurence_count
from utils.dist_util import compute_intersect_count, compute_wordnet_similarity, compute_tfidf_score

class FeatureExtraction(object):

	THRESHOLD = 2.0
	MATCH_THRESHOLD = 0.8

	def __init__(self, preprocessor):
		self.data = preprocessor.data
		self.description = preprocessor.description
		self.attributes = preprocessor.attributes
		self.product_title_mapping = preprocessor.product_title_map
		self.word2Vec_model_fname = preprocessor.word2Vec_model_fname
		self.doc2Vec_model_fname = preprocessor.doc2Vec_model_fname
		self.build_attribute_dict()

	def convert_to_label(self, x):
		if x <= 1.67:
			return 1
		elif x <= 2.5:
			return 2
		else:
			return 3

	def extractContextualFeatures(self):
		self.y = map(lambda x: 1 if x > FeatureExtraction.THRESHOLD else 0, self.data['relevance'])
		self.y_ternary = map(lambda x: self.convert_to_label(x), self.data['relevance'])
		self.word2Vec_model = gensim.models.Word2Vec.load(self.word2Vec_model_fname)
		self.doc2Vec_model = gensim.models.Doc2Vec.load(self.doc2Vec_model_fname)
		#self.construct_wordnet_features()
		self.construct_tfidf_features()
		self.construct_jaccard_coef_features()
		self.construct_edit_distance_features()
		self.construct_intersect_count_features()
		self.construct_coccurence_count_features()
		self.construct_non_contextual_features()
		self.construct_word2vec_features()
		self.construct_atrribute_features()
		self.construct_string_match_features()
		self.construct_doc2vec_features()

	def construct_jaccard_coef_features(self):
		print "constructing jaccard coefficient related features .... "
		t1 = time.time()
		self.jaccard_unigram_title = self.data.apply(lambda x: compute_jaccard(x['product_title'], x['search_term'], ngram = 1), 
									axis = 1)
		self.jaccard_unigram_desc = self.data.apply(lambda x: compute_jaccard(self.description[x['product_uid']], x['search_term'], 
									ngram = 1), axis = 1)

		self.jaccard_bigram_title = self.data.apply(lambda x: compute_jaccard(x['product_title'], x['search_term'], ngram = 2), 
									axis = 1)
		self.jaccard_bigram_desc = self.data.apply(lambda x: compute_jaccard(self.description[x['product_uid']], x['search_term'], 
									ngram = 2), axis = 1)
		print "Time while constructing jaccard coefficient related features is %d seconds \n" % int(time.time() - t1)	

	def construct_edit_distance_features(self):
		print "constructing edit distance related features .... "
		t1 = time.time()
		self.edit_distance_title = self.data.apply(lambda x: compute_edit_distance(x['product_title'], x['search_term']), axis = 1)
		self.edit_distance_desc = self.data.apply(lambda x: compute_edit_distance(self.description[x['product_uid']], 
									x['search_term']), axis = 1)
		print "Time while constructing edit distance related features is %d seconds \n" % int(time.time() - t1)	
	
	def construct_intersect_count_features(self):
		print "constructing intersect count related features .... "
		t1 = time.time()
		self.first_intersect_count_unigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'],
											x['search_term'], 0, 1, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.last_intersect_count_unigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], 
											x['search_term'], -1, 1, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.first_intersect_count_bigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], 
											x['search_term'], 0, 2, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.last_intersect_count_bigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], 
											x['search_term'], -1, 2, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.intersect_count = self.data.apply(lambda x: compute_intersect_count(x['product_title'], x['search_term'], 
								1, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		print "Time while constructing intersect count related features is %d seconds \n" % int(time.time() - t1)	
	
	def construct_coccurence_count_features(self):
		print "constructing coccurence related features .... "
		t1 = time.time()
		self.coccurence_count = self.data.apply(lambda x: compute_coccurence_count(x['product_title'], x['search_term'], 
								1, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		print "Time while constructing coccurence related features is %d seconds \n" % int(time.time() - t1)	

	def construct_non_contextual_features(self):
		print "constructing non contextual features .... "
		t1 = time.time()
		self.attr_bullet_ratio = self.data.apply(lambda x:self.bullet_ratio[x['product_uid']] if x['product_uid'] in self.bullet_ratio 
								else 0.0, axis = 1)
		self.attr_has_height = self.data.apply(lambda x:self.has_height[x['product_uid']] if x['product_uid'] in self.has_height 
								else 0.0, axis = 1)
		self.attr_has_depth = self.data.apply(lambda x:self.has_depth[x['product_uid']] if x['product_uid'] in self.has_depth else 0.0,
								axis = 1)
		self.attr_has_length = self.data.apply(lambda x:self.has_length[x['product_uid']] if x['product_uid'] in self.has_length 
								else 0.0, axis = 1)
		self.attr_has_width = self.data.apply(lambda x:self.has_width[x['product_uid']] if x['product_uid'] in self.has_width else 0.0,
								axis = 1)
		print "Time while constructing non contextual features is %d seconds \n" % int(time.time() - t1)	

	def construct_word2vec_features(self):
		print "constructing features from word2vec .... "
		t1 = time.time()
		# TODO(akshay) - Try also with product description
		self.avg_similarity = self.data.apply(lambda x: compute_cosine_similarity(x['product_title'], x['search_term'], 
								self.word2Vec_model), axis = 1)
		self.rmse_title = self.data.apply(lambda x:compute_rmse(x['product_title'], x['search_term'], self.word2Vec_model), axis = 1)
		self.rmse_desc = self.data.apply(lambda x:compute_rmse(self.description[x['product_uid']], x['search_term'], 
						self.word2Vec_model), axis = 1)
		print "Time while constructing features from word2vec is %d seconds \n" % int(time.time() - t1)	

	def construct_string_match_features(self):
		print "constructing string matching related features .... "
		t1 = time.time()
		self.longest_match = self.data.apply(lambda x:compute_longest_match(x['product_title'], x['search_term']), axis = 1)
		print "Time while constructing string matching related features is %d seconds \n" % int(time.time() - t1)	

	def construct_atrribute_features(self):
		print "constructing features from attributes .... "
		t1 = time.time()
		self.match_attr_ratio = self.data.apply(lambda x:compute_match_attr_ratio(x['product_uid'], x['search_term'], 
								self.attribute_dict), axis = 1)
		print "Time while constructing features from attributes is %d seconds \n" % int(time.time() - t1)	
	
	def construct_doc2vec_features(self):
		print "constructing features from doc2vec .... "
		t1 = time.time()
		self.doc2vec_rmse_title = self.data.apply(lambda x:compute_rmse(x['product_title'], x['search_term'], self.doc2Vec_model), 
								axis = 1)
		self.doc2vec_rmse_desc = self.data.apply(lambda x:compute_rmse(self.description[x['product_uid']], x['search_term'], 
						self.doc2Vec_model), axis = 1)
		print "Time while constructing features from doc2vec is %d seconds \n" % int(time.time() - t1)	

	def construct_wordnet_features(self):
		print "constructing wordnet features .... "
		t1 = time.time()
		self.path_wordnet_similarity = self.data.apply(lambda x: compute_wordnet_similarity(x['product_title'], x['search_term']),
										axis = 1)
		print "Time while constructing features from doc2vec is %d seconds \n" % int(time.time() - t1)

	def construct_tfidf_features(self):
		print "constructing tfidf features .... "
		t1 = time.time()
		self.tf_idf_ = TfidfVectorizer()
		self.X_ = self.data.apply(lambda x: x['product_title'] + self.description[x['product_uid']], axis = 1)
		self.dt_matrix = self.tf_idf_.fit_transform(self.X_)
		self.dt_matrix_dense = self.dt_matrix.toarray()
		self.tf_idf_scores = np.zeros(self.data.shape[0])
		for index, row in self.data.iterrows():
			self.tf_idf_scores[index] = compute_tfidf_score(self.dt_matrix_dense[index], self.tf_idf_,row['search_term'])
		print "Time while constructing tfidf features is %d seconds \n" % int(time.time() - t1)

	def extractNonContextualFeatures(self):
		self.bullet_ratio = {}
		self.has_height = {}
		self.has_depth = {}
		self.has_length = {}
		self.has_width = {}

		for product, attributes in self.attribute_dict.iteritems():
			self.bullet_ratio[product], self.has_height[product], self.has_depth[product] = 0.0, 0, 0
			self.has_length[product], self.has_width[product] = 0, 0
			for k, v in attributes.iteritems():
				if k.startswith("bullet"):
					self.bullet_ratio[product] = self.bullet_ratio[product] + 1
				elif k.find("height") >= 0:
					self.has_height[product] = 1
				elif k.find("depth") >= 0:
					self.has_depth[product] = 1
				elif k.find("width") >= 0:
					self.has_width[product] = 1
				elif k.find("length") >= 0:
					self.has_length[product] = 1
			self.bullet_ratio[product] = (self.bullet_ratio[product] * 1.0) / len(attributes.keys())

	def combineFeatures(self):
		self.features = zip(self.jaccard_unigram_title, self.jaccard_unigram_desc, self.jaccard_bigram_title, self.jaccard_bigram_desc,
						self.edit_distance_title, self.edit_distance_desc,
						self.first_intersect_count_unigram, self.last_intersect_count_unigram,
						self.first_intersect_count_bigram, self.last_intersect_count_bigram, self.intersect_count,
						self.avg_similarity, self.rmse_title, self.rmse_desc,
						self.longest_match,
						self.match_attr_ratio, 
						self.coccurence_count,
						self.doc2vec_rmse_title, self.doc2vec_rmse_desc,
						#self.path_wordnet_similarity,
						self.tf_idf_scores,
						self.attr_bullet_ratio, self.attr_has_height, self.attr_has_depth, self.attr_has_length, self.attr_has_width,
						self.y,
						self.y_ternary,
						self.data['relevance'])

	def build_attribute_dict(self):
		self.attribute_dict = {}
		for idx, row in self.attributes.iterrows():
			if row['product_uid'] not in self.attribute_dict:
				self.attribute_dict[row['product_uid']] = dict()
			self.attribute_dict[row['product_uid']][row['name']] = row['value']
		