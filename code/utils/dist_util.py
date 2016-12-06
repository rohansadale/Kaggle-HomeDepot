# -*- coding: utf-8 -*-
import numpy as np
from difflib import SequenceMatcher
from Levenshtein import distance
from nltk.corpus import wordnet as wn

"""
Function to compute normalized edit distance between two strings.
Edit distance => https://en.wikipedia.org/wiki/Edit_distance
example edit_distance('kitten', 'sitting') => 3

Edit distance is normalized so we compare values on same scale. For big strings
value might be big and for smaller string it might be smaller. So we need to account
for variation of string length.
"""
def compute_edit_distance(s1, s2, should_normalize = True):
	s1, s2 = str(s1), str(s2)
	dist = distance(s1, s2)
	if should_normalize:
		dist = dist / float(max(len(s1),len(s2)))
		dist = round(dist, 3)
	return dist

"""
Function to compute Jaccard Coefficient between two strings. We split the strings into n-grams
and then compute Jaccard Coefficient of two sets thus obtained.
Jaccard Coefficient => https://en.wikipedia.org/wiki/Jaccard_index
"""
def compute_jaccard(s1, s2, ngram):
	A = set(gen_ngrams(s1, ngram))
	B = set(gen_ngrams(s2, ngram))
	return _try_divide(float(len(A.intersection(B))) , len(A.union(B)))

"""
Function to find how many word ngram in target(search term;s2) closely matches with first ngram
in observation(product title;s1). Threshold denotes how relaxation we are allowing. Threshold = 1.0
implies we are looking for perfect match. Again we are computing normalizing to account different length
of string. Similarily, we are also finding how many word ngram in target(search term;s2) closely matches 
with last ngram in observation(product title;s1). This feature should give more weight to first and last ngram.
"""
def compute_first_last_intersect(s1, s2, idx, ngram, threshold):
	s1_ngrams = gen_ngrams(s1, ngram)
	s2_ngrams = gen_ngrams(s2, ngram)
	if 0 == len(s1_ngrams):
		return 0.0
	ct = 0

	for entry in s2_ngrams:
		if is_str_match(entry, s1_ngrams[idx], threshold):
			ct = ct + 1
	return _try_divide(float(ct), len(s2_ngrams))

"""
function to check if two strings matches with given level of similarity.
"""
def is_str_match(src, target, threshold):
	if threshold == 1.0:
		return src == target
	return (1.0 - compute_edit_distance(src, target)) >= threshold

"""
Function to compute how close are search term and product title. For each word in search_term, compute the
maximum value of the cosine similarity between this word and every other word in product title; then take
the minimum value of those maximum values as output feature.
"""
def compute_cosine_similarity(target, src, model):
	target_tokens = [w for w in target.split(" ") if w in model]
	src_tokens = [w for w in src.split(" ") if w in model]
	if 0 == len(target_tokens) or 0 == len(src_tokens):
		return 0.0
	min_similarity = 2.0
	for src_entry in src_tokens:
		max_similarity = max(map(lambda x: cosine_similarity(model[src_entry], model[x]), target_tokens))
		min_similarity = min(min_similarity, max_similarity)
	return min_similarity if min_similarity != 2.0 else 0.0

"""
Function to compute longest matching between two strings normalized by string length.
More Reference:- https://pymotw.com/2/difflib/
"""
def compute_longest_match(target, src):
	sq = SequenceMatcher(lambda x: x==" ", target, src)
	match = sq.find_longest_match(0, len(target), 0, len(src))
	return _try_divide(match.size, min(len(target), len(src)))

"""
RMSE between centroid vector of search_term and product title.
"""
def compute_rmse(src, target, model):
	target_tokens = [w for w in target.split(" ") if w in model]
	src_tokens = [w for w in src.split(" ") if w in model]
	
	dim = model.vector_size
	src_centroid = np.zeros(dim)
	target_centroid = np.zeros(dim)

	for entry in src_tokens:
		src_centroid = src_centroid + np.array(model[entry])
	if len(src_tokens) > 0:
		src_centroid = src_centroid / (len(src_tokens) * 1.0)

	for entry in target_tokens:
		target_centroid = target_centroid + np.array(model[entry])
	if len(target_tokens) > 0:
		target_centroid = target_centroid / (len(target_tokens) * 1.0)

	return np.sum((src_centroid - target_centroid)**2) / (dim * 1.0)

"""
Function to compute cosine_similarity between two vectors.
"""
def cosine_similarity(a, b):
	csum = 0.0
	for i, entry in enumerate(a):
		csum = csum + a[i] * b[i]
	return csum

"""
Function to count of attribute in product_attribute_list that matches search_term. Note that we don't have
attributes for every product.
"""
def compute_match_attr_ratio(product_id, search_term, attributes):
	if product_id not in attributes:
		return 0.0
	attribute_dict = attributes[product_id]
	search_term_tokens = set(search_term.split(" "))
	ans = 0
	for k, v in attribute_dict.iteritems():
		key_tokens = set(k.split(" "))
		if len(search_term_tokens.intersection(key_tokens)) > 0:
			ans = ans + 1
	return ans * 1.0 / len(attribute_dict.keys())

"""
count of word ngram of search term that closely matches with any ngram of target
"""
def compute_intersect_count(s1, s2, ngram, threshold):
	s1_ngrams = gen_ngrams(s1, ngram)
	s2_ngrams = gen_ngrams(s2, ngram)
	if 0 == len(s2_ngrams):
		return 0.0
	ct = 0

	for search_term in s2_ngrams:
		for target_term in s1_ngrams:
			if is_str_match(search_term, target_term, threshold):
				ct = ct + 1
				break

	return _try_divide(float(ct), len(s2_ngrams))

"""
count of closely matching word ngram pairs between search term and target

# ----------------------------------------------------------------------------
# How many cooccurrence ngrams between obs and target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
# ->
# CooccurrenceCount: 7 (i.e., AB x 2 + AB x 2 + AB x 2 + AC x 1)
# CooccurrenceRatio: 7/(6 x 5)
"""
def compute_coccurence_count(s1, s2, ngram, threshold):
	s1_ngrams = gen_ngrams(s1, ngram)
	s2_ngrams = gen_ngrams(s2, ngram)
	if 0 == len(s1_ngrams) or 0 == len(s2_ngrams):
		return 0.0
	ct = 0

	for search_term in s2_ngrams:
		for target_term in s1_ngrams:
			if is_str_match(search_term, target_term, threshold):
				ct = ct + 1

	return _try_divide(float(ct), len(s1_ngrams) * len(s2_ngrams))

"""
Return a score denoting how similar two word senses are, based on the shortest 
path that connects the senses in the is-a (hypernym/hypnoym) taxonomy. The 
score is in the range 0 to 1, except in those cases where a path cannot be 
found (will only be true for verbs as there are many distinct verb taxonomies),
in which case -1 is returned. A score of 1 represents identity i.e. comparing
a sense with itself will return 1.
"""
def compute_wordnet_similarity(s1, s2):
	target_tokens = gen_ngrams(s1, 1)
	obs_tokens = gen_ngrams(s2, 1)
	if 0 == len(target_tokens) or 0 == len(obs_tokens):
		return 0.0
	
	obs_synset_list = [wn.synsets(obs_token) for obs_token in obs_tokens]
	target_synset_list = [wn.synsets(target_token) for target_token in target_tokens]

	result = 1.0
	for obs_synset in obs_synset_list:
		score_list = []
		for target_synset in target_synset_list:
			max_similarity = maximum_similarity_for_two_synset_list(obs_synset, target_synset)
			score_list.append(max_similarity)
		result = min(result, max(score_list))
	return result

def maximum_similarity_for_two_synset_list(syn_list1, syn_list2):
	max_score = 0.
	if syn_list1 and syn_list2:
		for syn1 in syn_list1:
			for syn2 in syn_list2:
				try:
					_s = wn.path_similarity(syn1, syn2)
				except:
					_s = -1
				if _s and _s > max_score:
					max_score = _s
	return max_score

def compute_tfidf_score(dt_matrix, tf_idf, query):
	tokens = query.split(" ")
	score = 0.0
	for entry in tokens:
		if entry in tf_idf.vocabulary_:
			score = score + dt_matrix[tf_idf.vocabulary_[entry]]
	return score

"""
Function to generate ngrams.
"""
def gen_ngrams(text, ngrams = 1):
	tokens = text.split(' ')
	output = []
	for i in range(len(tokens)-ngrams+1):
		output.append(tokens[i:i+ngrams])
	return [' '.join(x) for x in output]

def _try_divide(num, den):
	if 0 == den:
		return den
	return round(float(num) / den, 3)