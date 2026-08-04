import re
import yaml
from pathlib import Path

# --- Try to load spaCy for medical entity density (Option A) ---
try:
    import spacy
    # Important: explicitly import EntityLinker or "scispacy_linker" will not be registered.
    from scispacy.linking import EntityLinker
    from scispacy.abbreviation import AbbreviationDetector
    
    nlp = spacy.load("en_core_sci_lg")
    
    # Add UMLS linker.
    # Note: the first run may download ~500MB+ of UMLS resources via scispaCy.
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    print("INFO: Loaded en_core_sci_lg with UMLS Linker.")

except ImportError:
    # scispaCy not installed.
    nlp = None
    print("WARNING: scispacy or scispacy.linking not found. Information Density will use fallback.")
except Exception as e:
    # Model load failure or other initialization errors.
    nlp = None
    print(f"WARNING: Failed to load Spacy/Linker. Error: {e}")


try:
    CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "analysis_config.yaml"

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f).get('lexicons', {})
    UNCERTAINTY_PHRASES = config.get('uncertainty_phrases', [])
    UNCERTAINTY_WEIGHTS = config.get('uncertainty_weights', {})
    CERTAIN_PHRASES = config.get('certainty_phrases', [])
except FileNotFoundError:
    print("WARNING: analysis_config.yaml not found. Using empty lexicons for linguistic features.")
    UNCERTAINTY_WEIGHTS = {}
    CERTAIN_PHRASES = []


# --- Core clinical semantic types (whitelist) ---
# Only entities with these TUIs count toward the numerator.
VALID_TUIS = {
    "T047", # Disease or Syndrome
    "T048", # Mental or Behavioral Dysfunction
    "T184", # Sign or Symptom
    "T033", # Finding
    "T121", # Pharmacologic Substance
    "T109", # Organic Chemical
    "T195", # Antibiotic
    "T023", # Body Part, Organ, or Organ Component
    "T060", # Diagnostic Procedure
    "T061", # Therapeutic or Preventive Procedure
    "T059", # Laboratory Procedure
}

def calculate_redundancy_score(assistant_text: str, ground_truth_text: str) -> int:
    """
    Compute redundancy score: (total words - overlap with ground truth).
    Score is non-positive; smaller (more negative) means higher redundancy.
    """
    if not isinstance(assistant_text, str) or not isinstance(ground_truth_text, str):
        return 0
        
    assistant_words = set(re.findall(r'\b\w+\b', assistant_text.lower()))
    ground_truth_words = set(re.findall(r'\b\w+\b', ground_truth_text.lower()))
    
    overlap_length = len(assistant_words.intersection(ground_truth_words))
    redundancy = len(assistant_words) - overlap_length
    
    return -redundancy


def calculate_conciseness_ratio(assistant_text: str, ground_truth_text: str) -> float:
    """
    Reference-based conciseness.
    Formula: Unique_Overlap_Count / Total_Assistant_Tokens
    """
    if not isinstance(assistant_text, str) or not isinstance(ground_truth_text, str):
        return 0.0
        
    # 1) Get all assistant tokens (keep duplicates for denominator).
    assistant_tokens = re.findall(r'\b\w+\b', assistant_text.lower())
    total_tokens = len(assistant_tokens)
    
    if total_tokens == 0:
        return 0.0
        
    # 2) Get unique tokens from ground truth.
    gt_unique = set(re.findall(r'\b\w+\b', ground_truth_text.lower()))
    
    # 3) Numerator: how many unique GT concepts are covered by the assistant.
    # Using sets reflects concept coverage; "Flu Flu" still counts once.
    assistant_unique = set(assistant_tokens)
    overlap_count = len(assistant_unique.intersection(gt_unique))
    
    # 4) Compute score.
    # Numerator: captured concepts; denominator: total token cost.
    score = overlap_count / total_tokens if total_tokens > 0 else 0.0
    
    return float(min(1.0, score))


TOKEN_PATTERN = re.compile(r"\b\w+\b")
VS_PATTERN = re.compile(r"(?<!\w)vs(?!\w)", flags=re.IGNORECASE)

def _compile_phrase_pattern(phrases):
    phrases = [p.strip().lower() for p in phrases if isinstance(p, str) and p.strip()]
    # Remove "vs" regardless of spacing.
    phrases = [p for p in phrases if p.strip() != "vs"]
    # Prefer longer phrases to reduce overlap.
    phrases = sorted(set(phrases), key=len, reverse=True)
    if not phrases:
        return None
    pattern = r"(?<!\w)(" + "|".join(map(re.escape, phrases)) + r")(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)

HEDGE_PATTERN = _compile_phrase_pattern(UNCERTAINTY_PHRASES)

def calculate_linguistic_certainty_score(assistant_text: str, ground_truth_text: str) -> float:
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return float("nan")

    lower = assistant_text.lower()
    token_count = len(TOKEN_PATTERN.findall(lower))
    if token_count == 0:
        return float("nan")

    hedges = 0
    if HEDGE_PATTERN is not None:
        hedges += len(HEDGE_PATTERN.findall(lower))
    hedges += len(VS_PATTERN.findall(lower))  # Count "vs" as hedging.

    eps = 1
    hedge_density = hedges / (token_count + eps)
    return float(1.0 - min(1.0, hedge_density))



def calculate_uncertainty_count(assistant_text: str, ground_truth_text: str) -> int:
    """
    Simple uncertainty score based on counts.
    More uncertainty phrases => lower (more negative) score.
    (ground_truth_text is unused; kept for signature consistency.)
    """
    if not isinstance(assistant_text, str):
        return 0

    lower_text = assistant_text.lower()
    # Count total occurrences of phrases in the list.
    count = sum(1 for phrase in UNCERTAINTY_PHRASES if phrase.lower() in lower_text)
    
    return -count


def calculate_uncertainty_weighted(assistant_text: str, ground_truth_text: str) -> int:
    """
    Weighted uncertainty score.
    Score is non-positive; smaller (more negative) means higher uncertainty.
    (ground_truth_text is unused; kept for signature consistency.)
    """
    if not isinstance(assistant_text, str):
        return 0

    lower_text = assistant_text.lower()
    score = 0
    for phrase, weight in UNCERTAINTY_WEIGHTS.items():
        if phrase in lower_text:
            score -= weight
    return score


def calculate_certainty_score(assistant_text: str, ground_truth_text: str) -> int:
    """
    Certainty score.
    Score is non-negative; higher means more certain.
    (ground_truth_text is unused; kept for signature consistency.)
    """
    if not isinstance(assistant_text, str):
        return 0
        
    lower_text = assistant_text.lower()
    score = 0
    for phrase in CERTAIN_PHRASES:
        if phrase in lower_text:
            score += 1
    return score


def calculate_information_density(assistant_text: str, ground_truth_text: str = None) -> float:
    """
    Information density (strict version).
    Defined as: valid tokens within medical entities / total valid tokens.
    
    "Valid tokens" are those that are not punctuation and not whitespace.
    This removes punctuation/formatting effects from the score.
    
    Returns a float in [0.0, 1.0].
    """
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return 0.0

    # Option A: use spaCy (recommended).
    doc = nlp(assistant_text)
    
    # 1) Denominator: count of all valid tokens.
    valid_tokens = [t for t in doc if not t.is_punct and not t.is_space]
    total_valid_count = len(valid_tokens)
    
    if total_valid_count == 0:
        return 0.0
        
    # 2) Numerator: valid tokens that belong to entities.
    # Use a set of indices to guard against potential overlaps.
    entity_token_indices = set()
    for ent in doc.ents:
        for token in ent:
            entity_token_indices.add(token.i)
    
    # Count tokens that are both valid and inside entities.
    medical_valid_count = sum(1 for t in valid_tokens if t.i in entity_token_indices)
    
    # 3) Density.
    density = medical_valid_count / total_valid_count
    return density
    

def calculate_umls_concept_density(assistant_text: str, ground_truth_text: str = None) -> float:
    """
    Core clinical concept density.
    Use the UMLS linker to filter broad medical terms and keep only
    diseases, symptoms, drugs, anatomy, and other core concepts.
    """
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return float("nan")

    if not nlp:
        return float("nan")

    try:
        doc = nlp(assistant_text)
        
        # Denominator: valid tokens (not punctuation, not whitespace).
        valid_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        total_valid_count = len(valid_tokens)
        
        if total_valid_count == 0:
            return float("nan")
            
        # Numerator: valid tokens that belong to core semantic types.
        core_entity_token_indices = set()

        linker = nlp.get_pipe("scispacy_linker")
        for ent in doc.ents:
            # ent._.kb_ents contains UMLS concepts (CUI, score).
            # Use the top-1 match.
            if len(ent._.kb_ents) > 0:
                # Get best-match CUI and entity.
                cui, score = ent._.kb_ents[0]
                
                # kb_entity = linker.kb.cui_to_entity[cui]
                kb_entity = linker.kb.cui_to_entity.get(cui)
                if kb_entity is None:
                    continue

                # Check if the concept's TUI is in the whitelist.
                # kb_entity.types is a list of TUIs, e.g. ['T047', 'T184'].
                if any(tui in VALID_TUIS for tui in kb_entity.types):
                    # If matched, add all token indices of this entity.
                    for token in ent:
                        core_entity_token_indices.add(token.i)
        
        # Count numerator.
        medical_valid_count = sum(1 for t in valid_tokens if t.i in core_entity_token_indices)
        
        return float(medical_valid_count / total_valid_count)

    except Exception as e:
        print(f"Error in density calc: {e}")
        return float("nan")



# --- Registry: map feature names to calculators ---
# Enables FeatureEngineer to call calculators dynamically.
AVAILABLE_LINGUISTIC_CALCULATORS = {
    'ConceptDensity': calculate_umls_concept_density,
    'LingCert': calculate_linguistic_certainty_score,
}

if __name__ == "__main__":
    # reasoning_sample = "The patient presents with right lower quadrant pain and rebound tenderness, highly suggestive of acute appendicitis."
    # score = calculate_information_density(reasoning_sample)
    # print(f"Reasoning: {reasoning_sample}")
    # print(f"Information Density: {score:.4f}")
    print(calculate_umls_concept_density("appendicitis vs gastroenteritis", nlp))
