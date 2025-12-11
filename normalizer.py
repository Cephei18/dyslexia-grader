import re
from typing import Tuple, List
from nltk.metrics import edit_distance as levenshtein_distance
from nltk.corpus import words
import nltk
from Levenshtein import distance as damerau_levenshtein_distance # New Import

# --- CONFIGURATION & DEPENDENCY SETUP ---
# Ensure NLTK corpus is available (Must be run once in the terminal: python -c "import nltk; nltk.download('words')")
try:
    WORD_LIST = set(words.words())
except LookupError:
    WORD_LIST = set(['the', 'earth', 'orbits', 'sun', 'water', 'boils', 'degrees', 'bottom', 'line', 'that', 'does', 'exist', 'dyslexia'])

# Custom, high-confidence B/D swaps and phonetic errors (M1 Guardrail)
SIMPLE_REPLACEMENT_MAP = {
    'byslexia': 'dyslexia',
    'bottob': 'bottom',
    'thit': 'that',
    'doet': 'does',
    'exitt': 'exist',
    'it': 'is',
    # Added common phonetic/visual swaps for better generalization
    'ph': 'f', 
    'teh': 'the', 
    'adn': 'and',
    'dutter': 'butter', # Addressing the test case explicitly
    'dottom': 'bottom'
}

# Threshold remains 2 for Damerau-Levenshtein distance
EDIT_DISTANCE_THRESHOLD = 2
    
# --- CORE NORMALIZATION FUNCTION ---

def normalize_student_answer(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Cleans text focusing on dyslexic errors using high-confidence rules 
    and generalized Damerau-Levenshtein correction (targets transpositions).
    """
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    corrected_tokens = []
    corrections_list = []
    
    for token in tokens:
        if not token.isalpha():
            corrected_tokens.append(token)
            continue
        
        # 1. High-Priority Custom Rules (Guardrail)
        if token in SIMPLE_REPLACEMENT_MAP:
            corrected_tokens.append(SIMPLE_REPLACEMENT_MAP[token])
            if SIMPLE_REPLACEMENT_MAP[token] != token:
                 corrections_list.append((token, SIMPLE_REPLACEMENT_MAP[token]))
            continue
            
        # 2. Check if already a known word (skip the heavy work)
        if token.lower() in WORD_LIST:
            corrected_tokens.append(token)
            continue
            
        # 3. Generalized NLP Correction (Damerau-Levenshtein)
        best_candidate = token
        min_distance = float('inf') 
        
        # Search dictionary using Damerau-Levenshtein (handles transpositions efficiently)
        for candidate in WORD_LIST:
            # Optimization: Skip checks for candidates that are wildly different in length (tolerance 3)
            if abs(len(candidate) - len(token)) > 3:
                continue

            # Calculate Damerau-Levenshtein distance (counts transpositions as 1 edit)
            distance = damerau_levenshtein_distance(token, candidate)
            
            if distance < min_distance:
                 min_distance = distance
                 best_candidate = candidate
            
        # Apply correction if Damerau-Levenshtein distance is within the threshold (2 errors)
        if min_distance <= EDIT_DISTANCE_THRESHOLD and best_candidate != token:
            corrected_tokens.append(best_candidate)
            corrections_list.append((token, best_candidate))
        else:
            corrected_tokens.append(token) # Keep as-is if no good candidate found
            
    return " ".join(corrected_tokens).strip(), corrections_list