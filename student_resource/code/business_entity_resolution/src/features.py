import os
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from rapidfuzz import fuzz

def score_candidate_pair_fast(
    n1, tok1, ft1, a1, nums1, post1,
    n2, tok2, ft2, a2, nums2, post2,
    is_s2
):
    """
    Multi-Channel Decision Engine for Entity Resolution.
    Returns (is_match: bool, score: float, features: list)
    """
    # 1. Exact Name & Address Shortcut
    if n1 and n1 == n2:
        if not a1 or not a2 or a1 == a2:
            return True, 0.99
            
    # 2. Fast Overlap Checks
    has_tok_overlap = bool(tok1 and tok2 and bool(tok1.intersection(tok2)))
    has_num_overlap = bool(nums1 and nums2 and bool(nums1.intersection(nums2)))
    has_post_overlap = bool(post1 and post2 and bool(post1.intersection(post2)))
    
    # 3. Name Similarities
    name_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0
    
    # Check Channel 1: Name match with NULL / Empty Address
    if not a2 or not a1:
        if name_sort_ratio >= 0.72 or name_set_ratio >= 0.85:
            return True, max(name_sort_ratio, name_set_ratio)
        if len(n1) >= 4 and len(n2) >= 4:
            if n1 in n2 or n2 in n1:
                return True, 0.80
        return False, 0.0

    # 4. Address Similarities (when address is present in both)
    addr_sort_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0
    
    # Check Channel 2: Strong Address Match with Brand Alias
    if has_num_overlap and addr_sort_ratio >= 0.72:
        return True, addr_sort_ratio
    if has_post_overlap and addr_sort_ratio >= 0.75:
        return True, addr_sort_ratio
    if addr_sort_ratio >= 0.88:
        return True, addr_sort_ratio
        
    # Check Channel 3: Dual Name + Address Match
    if name_sort_ratio >= 0.50 and addr_sort_ratio >= 0.50:
        return True, (name_sort_ratio + addr_sort_ratio) / 2.0
    if name_sort_ratio * addr_sort_ratio >= 0.28:
        return True, name_sort_ratio * addr_sort_ratio
    if name_set_ratio >= 0.75 and addr_sort_ratio >= 0.45:
        return True, 0.70
    if name_sort_ratio >= 0.75 and addr_sort_ratio >= 0.40:
        return True, 0.70
        
    return False, 0.0

def extract_pairwise_features_fast(
    n1, tok1, ft1, a1, nums1, post1,
    n2, tok2, ft2, a2, nums2, post2,
    is_s2
):
    name_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    
    tok_jaccard = len(tok1.intersection(tok2)) / len(tok1.union(tok2)) if (tok1 and tok2) else 0.0
    first_tok_match = 1.0 if (ft1 and ft2 and ft1 == ft2) else 0.0
    len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    
    if not a2:
        addr_missing = 1.0
        addr_sort_ratio = 0.0
        addr_set_ratio = 0.0
        addr_ratio = 0.0
        addr_partial = 0.0
        num_jaccard = 0.0
        postal_match = 0.0
    else:
        addr_missing = 0.0
        addr_sort_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
        num_jaccard = len(nums1.intersection(nums2)) / len(nums1.union(nums2)) if (nums1 and nums2) else 0.0
        postal_match = 1.0 if (post1 and post2 and bool(post1.intersection(post2))) else 0.0
        
    max_sim = max(name_sort_ratio, addr_sort_ratio)
    prod_sim = name_sort_ratio * addr_sort_ratio
    
    return [
        name_sort_ratio, name_set_ratio, name_ratio, name_partial,
        tok_jaccard, first_tok_match, len_diff,
        addr_missing, addr_sort_ratio, addr_set_ratio, addr_ratio, addr_partial,
        num_jaccard, postal_match,
        max_sim, prod_sim, float(is_s2)
    ]

FEATURE_NAMES = [
    'name_sort_ratio', 'name_set_ratio', 'name_ratio', 'name_partial',
    'tok_jaccard', 'first_tok_match', 'len_diff',
    'addr_missing', 'addr_sort_ratio', 'addr_set_ratio', 'addr_ratio', 'addr_partial',
    'num_jaccard', 'postal_match',
    'max_sim', 'prod_sim', 'is_s2'
]
