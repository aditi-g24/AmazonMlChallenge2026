import os
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gc
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODELS_DIR
from preprocessor import clean_record_fast
from blocking import MultiKeyBlocker
from features import score_candidate_pair_fast

def process_test_country_threaded(country, num_threads=8):
    print(f"\n=======================================================", flush=True)
    print(f"[*] Processing Country: {country} (High-Speed ThreadPool {num_threads} Threads)", flush=True)
    print(f"=======================================================", flush=True)
    t0 = time.time()
    
    # 1. Load Test S1 for this country
    test_s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s1_all = pd.read_csv(test_s1_path, sep="\t")
    s1_df = s1_all[s1_all['country'] == country].reset_index(drop=True)
    del s1_all
    gc.collect()
    
    n_s1 = len(s1_df)
    print(f"Loaded {n_s1} S1 records for {country} in {time.time()-t0:.2f}s", flush=True)
    
    # 2. Load Targets (S2 + S3) for this country
    t0 = time.time()
    s2_df = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t")
    s2_c = s2_df[s2_df['country'] == country]
    del s2_df
    
    s3_df = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t")
    s3_c = s3_df[s3_df['country'] == country]
    del s3_df
    gc.collect()
    
    all_targets = pd.concat([s2_c, s3_c], ignore_index=True)
    del s2_c, s3_c
    gc.collect()
    
    target_ids = all_targets['entity_id'].values
    target_names = all_targets['business_name'].values
    target_addrs = all_targets['business_address'].values
    del all_targets
    gc.collect()
    
    print(f"Loaded {len(target_ids)} target records for {country} in {time.time()-t0:.2f}s", flush=True)
    
    # 3. Fit Blocker Index
    t0 = time.time()
    blocker = MultiKeyBlocker(country=country)
    blocker.fit(target_ids, target_names, target_addrs)
    print(f"Blocker index built in {time.time()-t0:.2f}s ({len(blocker.idx_map)} keys)", flush=True)
    
    # 4. Process S1 Records with ThreadPool
    s1_data = list(zip(s1_df['entity_id'].values, s1_df['business_name'].values, s1_df['business_address'].values))
    del s1_df
    gc.collect()
    
    def _process_item(item):
        s1_id, s1_name, s1_addr = item
        s1_rec = clean_record_fast(s1_name, s1_addr, country)
        cand_indices = blocker.query(s1_rec, bucket_cap=45)
        
        n1 = s1_rec['name_clean']
        tok1 = s1_rec['name_tokens']
        ft1 = s1_rec['first_token']
        a1 = s1_rec['addr_clean']
        nums1 = s1_rec['nums']
        post1 = s1_rec['postals']
        
        matches = []
        cands = []
        
        for c_idx in cand_indices:
            tid = target_ids[c_idx]
            cands.append(tid)
            
            is_m, sc = score_candidate_pair_fast(
                n1, tok1, ft1, a1, nums1, post1,
                blocker.target_clean_names[c_idx], blocker.target_name_tokens[c_idx], blocker.target_first_tok[c_idx],
                blocker.target_clean_addrs[c_idx], blocker.target_nums[c_idx], blocker.target_postals[c_idx],
                blocker.target_is_s2[c_idx]
            )
            
            if is_m:
                matches.append(tid)
                
        return (s1_id, ",".join(matches)), (s1_id, ",".join(cands))
        
    print(f"Running multi-threaded inference over {n_s1} S1 entities...", flush=True)
    t_inf = time.time()
    all_matching = []
    all_candidates = []
    
    BATCH_SIZE = 50000
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for batch_start in range(0, n_s1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_s1)
            batch_items = s1_data[batch_start:batch_end]
            
            results = list(executor.map(_process_item, batch_items, chunksize=500))
            for m_tuple, c_tuple in results:
                all_matching.append(m_tuple)
                all_candidates.append(c_tuple)
                
            elapsed = time.time() - t_inf
            speed = batch_end / max(elapsed, 0.001)
            print(f"  Processed {batch_end}/{n_s1} S1 entities ({(batch_end/n_s1)*100:.1f}%) in {elapsed:.1f}s ({speed:.0f} S1/sec)", flush=True)
            
    print(f"[+] Finished {country} in {(time.time()-t_inf)/60:.2f} minutes!", flush=True)
    
    del blocker, target_ids, target_names, target_addrs, s1_data
    gc.collect()
    
    return all_matching, all_candidates
