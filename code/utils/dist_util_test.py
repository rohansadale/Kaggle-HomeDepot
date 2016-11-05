import sys
sys.path.append(".")
from utils.dist_util import compute_jaccard, compute_edit_distance, compute_first_last_intersect, compute_cosine_similarity
from utils.dist_util import compute_longest_match, compute_rmse, compute_match_attr_ratio

s1 = 'This is liverpool football club'
s2 = 'liverpool is good club'

assert(compute_edit_distance(s1,s2) == 0.516)
assert(compute_jaccard(s1, s2, 1) == )
assert(compute_jaccard(s1, s2, 2) == )
