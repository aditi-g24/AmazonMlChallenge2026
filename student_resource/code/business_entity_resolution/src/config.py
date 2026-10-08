import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Expanded Legal & Entity Suffixes (US, India, France)
LEGAL_SUFFIXES = {
    # US / UK / Global
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'co', 'company', 'lp', 'llp', 'pllc', 'gmbh', 'sa', 'ag', 'bv', 'nv',
    'holdings', 'holding', 'group', 'services', 'enterprises', 'solutions',
    'international', 'intl', 'global', 'technologies', 'tech', 'systems',
    'associates', 'consulting', 'management', 'logistics', 'ventures',
    
    # India
    'pvt', 'private', 'pvtltd', 'privatelimited', 'opc', 'm/s', 'ms',
    'enterprises', 'traders', 'trading', 'industries', 'stores', 'store',
    'agency', 'agencies', 'works', 'mart', 'bazaar', 'dhaba', 'hotel',
    'textiles', 'jewellers', 'jewelers', 'chemists', 'pharmacy',
    
    # France
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'ei', 'eirl', 'gie',
    'association', 'assoc', 'ste', 'sté', 'soc', 'societe', 'société',
    'ets', 'etablissement', 'etablissements', 'cabinet', 'atelier',
    'boulangerie', 'pharmacie', 'garage', 'restaurant', 'coiffure',
    'auto-ecole', 'auto', 'immo', 'immobilier', 'conseil', 'batiment', 'btp'
}

FRENCH_STREET_WORDS = {
    'rue': 'r', 'boulevard': 'bd', 'bvd': 'bd', 'avenue': 'av', 'ave': 'av',
    'chemin': 'ch', 'allée': 'all', 'allee': 'all', 'place': 'pl', 'route': 'rte',
    'impasse': 'imp', 'cours': 'crs', 'quai': 'qu', 'faubourg': 'fbg',
    'passage': 'pas', 'square': 'sq', 'zone': 'za', 'zi': 'za', 'zac': 'za',
    'immeuble': 'imm', 'batiment': 'bat', 'residence': 'res', 'etage': 'etg',
    'saint': 'st', 'sainte': 'ste'
}

STREET_WORDS_US_IN = {
    'street': 'st', 'str': 'st', 'avenue': 'av', 'ave': 'av', 'road': 'rd',
    'drive': 'dr', 'lane': 'ln', 'boulevard': 'blvd', 'blvd': 'blvd', 'bvd': 'blvd',
    'court': 'ct', 'place': 'pl', 'circle': 'cir', 'parkway': 'pkwy',
    'highway': 'hwy', 'expressway': 'expy', 'suite': 'ste', 'ste': 'ste',
    'apartment': 'apt', 'apt': 'apt', 'building': 'bldg', 'bldg': 'bldg',
    'floor': 'fl', 'fl': 'fl', 'block': 'blk', 'blk': 'blk', 'sector': 'sec',
    'sec': 'sec', 'nagar': 'ngr', 'colony': 'col', 'marg': 'mrg', 'chowk': 'chk',
    'bazaar': 'bzr', 'market': 'mkt', 'cross': 'crs', 'main': 'mn'
}

STATE_MAP = {
    'california': 'ca', 'new york': 'ny', 'texas': 'tx', 'florida': 'fl',
    'illinois': 'il', 'pennsylvania': 'pa', 'ohio': 'oh', 'georgia': 'ga',
    'north carolina': 'nc', 'michigan': 'mi', 'new jersey': 'nj', 'virginia': 'va',
    'washington': 'wa', 'arizona': 'az', 'massachusetts': 'ma', 'tennessee': 'tn',
    'indiana': 'in', 'missouri': 'mo', 'maryland': 'md', 'wisconsin': 'wi',
    'colorado': 'co', 'minnesota': 'mn', 'south carolina': 'sc', 'alabama': 'al',
    'louisiana': 'la', 'kentucky': 'ky', 'oregon': 'or', 'oklahoma': 'ok',
    'connecticut': 'ct', 'utah': 'ut', 'iowa': 'ia', 'nevada': 'nv',
    'arkansas': 'ar', 'mississippi': 'ms', 'kansas': 'ks', 'new mexico': 'nm',
    'nebraska': 'ne', 'idaho': 'id', 'west virginia': 'wv', 'hawaii': 'hi',
    'new hampshire': 'nh', 'maine': 'me', 'montana': 'mt', 'rhode island': 'ri',
    'delaware': 'de', 'south dakota': 'sd', 'north dakota': 'nd', 'alaska': 'ak',
    'vermont': 'vt', 'wyoming': 'wy', 'district of columbia': 'dc'
}

MATCH_THRESHOLD = 0.52
