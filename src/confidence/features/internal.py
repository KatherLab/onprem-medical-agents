import numpy as np
import pandas as pd
from typing import List, Dict, Any

def majority_vote_decision(runs: List[Dict[str, Any]]) -> bool:
    """
    Majority vote over N runs' "decision" values.

    Args:
        runs (list): List of run result dictionaries.

    Returns:
        bool: True if more than half of the runs are True, else False.
    """
    if not runs:
        return False
        
    decisions = [r.get('effective_decision', r.get('decision', False)) for r in runs]
    true_count = sum(decisions)
    return true_count > len(decisions) / 2


def calculate_quantitative_features(runs: List[Dict[str, Any]], target_key: str = 'diagnosis_prob') -> pd.Series:
    """
    Generic quantitative feature calculation.
    Compute mean and standard deviation for the given key across runs.

    Args:
        runs (list): List of run result dictionaries.
        target_key (str): Field to summarize.
                          - Diagnosis: 'diagnosis_prob'
                          - Reasoning: 'reason_prob'

    Returns:
        pd.Series: Series with 'Mean' and 'Std'.
    """
    # Default return value.
    default_result = pd.Series([0.0, 0.0], index=['Mean', 'Std'])

    if not runs or not isinstance(runs, list):
        return default_result

    # Extract valid probability values (filter out None/NaN).
    # Some runs may have null probabilities; filtering is required.
    probabilities = [
        r.get(target_key) 
        for r in runs 
        if r.get(target_key) is not None
    ]

    if not probabilities:
        return default_result

    # Mean score.
    mean_score = float(np.mean(probabilities))
    
    # Standard deviation (stability). If only one sample, std = 0.
    std_score = float(np.std(probabilities)) if len(probabilities) > 1 else 0.0

    return pd.Series([mean_score, std_score], index=['Mean', 'Std'])
