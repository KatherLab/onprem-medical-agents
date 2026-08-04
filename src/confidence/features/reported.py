import numpy as np
import pandas as pd
from typing import Dict, List

def calculate_reported_confidence_features(runs: List[Dict]) -> pd.Series:
    """
    Compute features related to self-reported confidence.
    Currently returns the mean self-reported confidence score.
    """
    if not runs:
        return pd.Series([None], index=['Reported_Confidence_Score'])
    
    # Extract valid self-reported confidences (0-100).
    reported_confidences = [
        r.get('reported_confidence') 
        for r in runs 
        if r.get('reported_confidence') is not None
    ]

    if not reported_confidences:
        return pd.Series([None], index=['Reported_Confidence_Score'])
    
    # Compute mean and normalize to [0, 1].
    reported_score = np.mean(reported_confidences) / 100.0
    
    return pd.Series([reported_score], index=['Reported_Confidence_Score'])
