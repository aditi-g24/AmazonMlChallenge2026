import os
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from collections import defaultdict
from preprocessor import clean_record_fast

class MultiKeyBlocker:
    """
    High-Speed Multi-Key Inverted Index Blocker.
    """
    def __init__(self, country='US'):
        self.country = country
        self.idx_map = defaultdict(list)
        self.target_ids = None
        self.target_clean_names = None
        self.target_name_tokens = None
        self.target_first_tok = None
        self.target_clean_addrs = None
        self.target_nums = None
        self.target_postals = None
        self.target_is_s2 = None

    def fit(self, target_ids, target_names, target_addrs):
        self.target_ids = target_ids
        self.idx_map = defaultdict(list)
        
        n_targets = len(target_ids)
        self.target_clean_names = [None] * n_targets
        self.target_name_tokens = [None] * n_targets
        self.target_first_tok = [None] * n_targets
        self.target_clean_addrs = [None] * n_targets
        self.target_nums = [None] * n_targets
        self.target_postals = [None] * n_targets
        self.target_is_s2 = [1 if tid.startswith('S2-') else 0 for tid in target_ids]
        
        for i in range(n_targets):
            rec = clean_record_fast(target_names[i], target_addrs[i], self.country)
            self.target_clean_names[i] = rec['name_clean']
            self.target_name_tokens[i] = rec['name_tokens']
            self.target_first_tok[i] = rec['first_token']
            self.target_clean_addrs[i] = rec['addr_clean']
            self.target_nums[i] = rec['nums']
            self.target_postals[i] = rec['postals']
            
            for k in rec['keys']:
                self.idx_map[k].append(i)

    def query(self, s1_rec, bucket_cap=45):
        candidates = set()
        for k in s1_rec['keys']:
            bucket = self.idx_map.get(k)
            if bucket is not None:
                if len(bucket) <= bucket_cap:
                    candidates.update(bucket)
                else:
                    candidates.update(bucket[:bucket_cap])
        return candidates
