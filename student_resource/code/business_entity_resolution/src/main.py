import os
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import time
import subprocess
import pandas as pd
from config import (
    TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODELS_DIR
)
from pipeline import process_test_country_threaded

def main():
    print("==================================================================", flush=True)
    print("   Amazon ML Challenge 2026: Business Entity Resolution Pipeline  ", flush=True)
    print("==================================================================", flush=True)
    total_start = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Read Test S1 metadata
    test_s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    test_s1_df = pd.read_csv(test_s1_path, sep="\t")
    required_s1_ids = list(test_s1_df['entity_id'])
    test_countries = list(test_s1_df['country'].unique())
    print(f"Found test countries: {test_countries} with {len(required_s1_ids)} total S1 records.\n", flush=True)
    
    all_matching_results = {}
    all_candidate_results = {}
    
    # Process each country partition
    for country in test_countries:
        m_res, c_res = process_test_country_threaded(country, num_threads=8)
        for s1_id, match_str in m_res:
            all_matching_results[s1_id] = match_str
        for s1_id, cand_str in c_res:
            all_candidate_results[s1_id] = cand_str
            
    # 2. Format & Export Output TSV files
    print("\n--- STEP 2: Exporting Submission Files ---", flush=True)
    matching_rows = []
    candidate_rows = []
    for s1_id in required_s1_ids:
        matching_rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': all_matching_results.get(s1_id, "")
        })
        candidate_rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': all_candidate_results.get(s1_id, "")
        })
        
    matching_df = pd.DataFrame(matching_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    
    matching_out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    matching_df.to_csv(matching_out_path, sep="\t", index=False, encoding="utf-8")
    candidate_df.to_csv(candidate_out_path, sep="\t", index=False, encoding="utf-8")
    
    print(f"Wrote matching_results.tsv: {len(matching_df)} rows to {matching_out_path}", flush=True)
    print(f"Wrote candidate_pairs.tsv: {len(candidate_df)} rows to {candidate_out_path}", flush=True)
    
    # 3. Fast Validation
    print("\n--- STEP 3: Validating Outputs with validate_submission.py ---", flush=True)
    validator_path = os.path.join(os.path.dirname(OUTPUT_DIR), "utils", "validate_submission.py")
    if os.path.isfile(validator_path):
        val_cmd = [
            sys.executable,
            "-u",
            validator_path,
            "--matching", matching_out_path,
            "--candidate", "non_existent.tsv",
            "--test-dir", TEST_DIR
        ]
        result = subprocess.run(val_cmd, capture_output=True, text=True)
        print("Validator Output:\n" + result.stdout, flush=True)
        if result.stderr:
            print("Validator Errors:\n" + result.stderr, flush=True)
            
    print(f"\n[+] Pipeline completed successfully in {(time.time() - total_start)/60:.2f} minutes!", flush=True)

if __name__ == "__main__":
    main()
