import pandas as pd
import sys

from preprocessor.PreProcessor import PreProcessor
from featureExtraction.FeatureExtraction import FeatureExtraction

base_dir = '/Users/akshaykulkarni/Documents/study/ML/course/Kaggle-HomeDepot/data/'
input_file = base_dir + 'train.csv' #sys.argv[1]
descriptions_file = base_dir + 'product_descriptions.csv' #sys.argv[2]
attributes_file = base_dir + 'attributes.csv' #sys.argv[3]

print "Reading input files .... "
data = pd.read_csv(input_file)
description = pd.read_csv(descriptions_file).reset_index('product_uid')
description = dict(zip(description.product_uid, description.product_description))
attribute = pd.read_csv(attributes_file)

print "Starting PreProcessor .... "
preprocessor = PreProcessor(data, description, attribute)
preprocessor.transform()

print "Creating features .... "
featureExtraction = FeatureExtraction(preprocessor)
featureExtraction.extractNonContextualFeatures()
featureExtraction.extractContextualFeatures()
featureExtraction.combineFeatures()
features_df = pd.DataFrame(featureExtraction.features,
			columns = ['jaccard_unigram_title', 'jaccard_unigram_desc', 'jaccard_bigram_title', 'jaccard_bigram_desc',
			'edit_distance_title', 'edit_distance_desc', 'first_intersect_count_unigram', 'last_intersect_count_unigram',
			'first_intersect_count_bigram', 'last_intersect_count_bigram', 'avg_similarity','rmse_title','rmse_desc', 'longest_match',
			'attribute_matched', 'bullet_count', 'intersect_count','coccurence_count', 'has_height', 'has_depth', 'has_length', 
			'has_width', 'is_relevant'])
features_df.to_csv(base_dir + 'features.csv', sep = ",", index=False, float_format='%.3f')