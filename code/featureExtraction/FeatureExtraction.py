import gensim
import sys
sys.path.append("..")
from utils.dist_util import compute_jaccard, compute_edit_distance, compute_first_last_intersect, compute_cosine_similarity
from utils.dist_util import compute_longest_match, compute_rmse, compute_match_attr_ratio

class FeatureExtraction(object):

	THRESHOLD = 2.0
	MATCH_THRESHOLD = 1.0

	def __init__(self, preprocessor):
		self.data = preprocessor.data
		self.description = preprocessor.description
		self.attributes = preprocessor.attributes
		self.product_title_mapping = preprocessor.product_title_map
		self.word2Vec_model_fname = preprocessor.word2Vec_model_fname
		self.build_attribute_dict()

	def extractContextualFeatures(self):
		self.y = map(lambda x: 1 if x > FeatureExtraction.THRESHOLD else 0, self.data['relevance'])
		self.word2Vec_model = gensim.models.Word2Vec.load(self.word2Vec_model_fname)
		
		self.jaccard_unigram_title = self.data.apply(lambda x: compute_jaccard(x['product_title'], x['search_term'], ngram = 1), axis = 1)
		self.jaccard_unigram_desc = self.data.apply(lambda x: compute_jaccard(self.description[x['product_uid']], x['search_term'], 
									ngram = 1), axis = 1)

		self.jaccard_bigram_title = self.data.apply(lambda x: compute_jaccard(x['product_title'], x['search_term'], ngram = 2), axis = 1)
		self.jaccard_bigram_desc = self.data.apply(lambda x: compute_jaccard(self.description[x['product_uid']], x['search_term'], 
									ngram = 2), axis = 1)

		self.edit_distance_title = self.data.apply(lambda x: compute_edit_distance(x['product_title'], x['search_term']), axis = 1)
		self.edit_distance_desc = self.data.apply(lambda x: compute_edit_distance(self.description[x['product_uid']], x['search_term']),
								 axis = 1)

		self.first_intersect_count_unigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], x['search_term'],
											0, 1, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.last_intersect_count_unigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], x['search_term'],
		 									-1, 1, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.first_intersect_count_bigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], x['search_term'],
		 									0, 2, FeatureExtraction.MATCH_THRESHOLD), axis = 1)
		self.last_intersect_count_bigram = self.data.apply(lambda x: compute_first_last_intersect(x['product_title'], x['search_term'],
		 									-1, 2, FeatureExtraction.MATCH_THRESHOLD), axis = 1)

		# TODO(akshay) - Try also with product description
		self.avg_similarity = self.data.apply(lambda x: compute_cosine_similarity(x['product_title'], x['search_term'], 
								self.word2Vec_model), axis = 1)
		self.rmse_title = self.data.apply(lambda x:compute_rmse(x['product_title'], x['search_term'], self.word2Vec_model), axis = 1)
		self.rmse_desc = self.data.apply(lambda x:compute_rmse(self.description[x['product_uid']], x['search_term'], self.word2Vec_model), axis = 1)
		self.longest_match = self.data.apply(lambda x:compute_longest_match(x['product_title'], x['search_term']), axis = 1)
		self.match_attr_ratio = self.data.apply(lambda x:compute_match_attr_ratio(x['product_uid'], x['search_term'], self.attribute_dict),
								axis = 1)
		self.attr_bullet_ratio = self.data.apply(lambda x:self.bullet_ratio[x['product_uid']] if x['product_uid'] in self.bullet_ratio else 0.0,
								axis = 1)
		self.attr_has_height = self.data.apply(lambda x:self.has_height[x['product_uid']] if x['product_uid'] in self.has_height else 0.0,
								axis = 1)
		self.attr_has_depth = self.data.apply(lambda x:self.has_depth[x['product_uid']] if x['product_uid'] in self.has_depth else 0.0,
								axis = 1)
		self.attr_has_length = self.data.apply(lambda x:self.has_length[x['product_uid']] if x['product_uid'] in self.has_length else 0.0,
								axis = 1)
		self.attr_has_width = self.data.apply(lambda x:self.has_width[x['product_uid']] if x['product_uid'] in self.has_width else 0.0,
								axis = 1)

	
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
						self.edit_distance_title, self.edit_distance_desc, self.first_intersect_count_unigram, self.last_intersect_count_unigram,
						self.first_intersect_count_bigram, self.last_intersect_count_bigram, self.avg_similarity, self.rmse_title, 
						self.rmse_desc, self.longest_match, self.match_attr_ratio, self.attr_bullet_ratio, self.attr_has_height,
						self.attr_has_depth, self.attr_has_length, self.attr_has_width, self.y)

	def build_attribute_dict(self):
		self.attribute_dict = {}
		for idx, row in self.attributes.iterrows():
			if row['product_uid'] not in self.attribute_dict:
				self.attribute_dict[row['product_uid']] = dict()
			self.attribute_dict[row['product_uid']][row['name']] = row['value']
		