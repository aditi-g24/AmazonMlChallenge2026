import os
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import re
import unicodedata
from config import LEGAL_SUFFIXES, FRENCH_STREET_WORDS, STREET_WORDS_US_IN, STATE_MAP

# Pre-compiled regexes
RE_URL = re.compile(r'https?://|www\.')
RE_EXT = re.compile(r'\.(com|org|net|in|co|io|fr|us|biz|info|gov|edu|gouv)$')
RE_TOKENS = re.compile(r'[a-z0-9]+')
RE_POSTAL_IN = re.compile(r'\b[1-9][0-9]{5}\b')
RE_POSTAL_US_FR = re.compile(r'\b[0-9]{5}\b')

STOP_WORDS_ADDR = {
    'the', 'and', 'near', 'opposite', 'unit', 'apartment', 'apt', 'floor',
    'suite', 'ste', 'null', 'nan', 'le', 'la', 'de', 'du', 'des', 'et', 'en',
    'au', 'aux', 'dans', 'sur'
}

def normalize_text_fast(text):
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ascii')
    return text.lower().strip()

def clean_record_fast(name, addr, country='US'):
    # 1. Clean Name
    n_str = normalize_text_fast(name)
    n_str = RE_URL.sub('', n_str)
    n_str = RE_EXT.sub('', n_str)
    if n_str.startswith('@'):
        n_str = n_str[1:]
        
    n_tokens = RE_TOKENS.findall(n_str)
    n_filt = [t for t in n_tokens if t not in LEGAL_SUFFIXES]
    chosen_n = n_filt if n_filt else n_tokens
    
    clean_name = " ".join(chosen_n)
    name_tokens = set(chosen_n)
    first_token = chosen_n[0] if chosen_n else ""
    
    # 2. Clean Address
    a_str = normalize_text_fast(addr)
    a_tokens = RE_TOKENS.findall(a_str)
    
    norm_a = []
    for t in a_tokens:
        if country == 'France' and t in FRENCH_STREET_WORDS:
            norm_a.append(FRENCH_STREET_WORDS[t])
        elif t in STREET_WORDS_US_IN:
            norm_a.append(STREET_WORDS_US_IN[t])
        elif t in STATE_MAP:
            norm_a.append(STATE_MAP[t])
        else:
            norm_a.append(t)
            
    clean_addr = " ".join(norm_a)
    
    if country == 'India':
        postal_codes = RE_POSTAL_IN.findall(a_str)
    else:
        postal_codes = RE_POSTAL_US_FR.findall(a_str)
    postal_set = set(postal_codes)
    
    nums = [str(int(t)) for t in a_tokens if t.isdigit() and len(t) <= 8]
    num_set = set(nums)
    
    words = [t for t in norm_a if len(t) >= 3 and not t.isdigit() and t not in STOP_WORDS_ADDR]
    
    # 3. Multi-Channel Blocking Keys
    keys = []
    if chosen_n:
        # K1: Sorted name tokens
        keys.append("ns_" + "_".join(sorted(chosen_n)))
        
        # K2: Concatenated name
        concat_n = "".join(chosen_n)
        if len(concat_n) >= 4:
            keys.append("nc_" + concat_n)
            
        # K3: Token 2-combinations (pairs)
        if len(chosen_n) >= 2:
            s_u = sorted(set(chosen_n))
            for i in range(len(s_u)):
                for j in range(i + 1, min(i + 3, len(s_u))):
                    keys.append(f"n2_{s_u[i]}_{s_u[j]}")
                    
        # K4: Prefix 2 tokens
        if len(chosen_n) >= 2:
            keys.append(f"np_{chosen_n[0]}_{chosen_n[1]}")
        elif len(chosen_n) == 1 and len(chosen_n[0]) >= 3:
            keys.append(f"np_{chosen_n[0]}")
            
        # K5: Hybrid first token + postal code / house num
        if first_token:
            for pc in postal_codes[:1]:
                keys.append(f"npc_{first_token}_{pc}")
            for num in nums[:1]:
                keys.append(f"nn_{first_token}_{num}")
                
    # K6: Address Number + Street word
    if nums and words:
        for num in nums[:2]:
            for w in words[:2]:
                keys.append(f"aw_{num}_{w}")
                
    # K7: Postal code + Street word
    if postal_codes and words:
        for pc in postal_codes[:1]:
            for w in words[:2]:
                keys.append(f"apc_{pc}_{w}")
                
    return {
        'name_clean': clean_name,
        'name_tokens': name_tokens,
        'first_token': first_token,
        'addr_clean': clean_addr,
        'nums': num_set,
        'postals': postal_set,
        'keys': keys
    }
