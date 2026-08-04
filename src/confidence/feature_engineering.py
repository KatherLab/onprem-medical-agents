import pandas as pd
import numpy as np

from .features import internal, behavioral
from .features.expressed import AVAILABLE_LINGUISTIC_CALCULATORS

class FeatureEngineer:
    """
    Compute all confidence scores and related features from raw data.
    """
    def __init__(self, config: dict):
        """
        Initialize the FeatureEngineer.

        Args:
            config (dict): Full configuration loaded from YAML.
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        self.enable_reasoning = self.feature_config.get('enable_reasoning_analysis', False)
        
        print(f"FeatureEngineer initialized. Reasoning Analysis Enabled: {self.enable_reasoning}")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.

        Args:
            df (pd.DataFrame): Raw master DataFrame from DataLoader.

        Returns:
            pd.DataFrame: Processed DataFrame with all engineered features.
        """
        print("\n--- Running Feature Engineering Step ---")

        # ==============================================================================
        # 0) Data prep: extract texts
        # ==============================================================================
        # print("  - Extracting Diagnosis texts...")
        # df['text_diagnosis'] = df.apply(lambda row: safe_get_run_data(row, key='assistant_diagnosis'), axis=1)
        
        # # Only extract Reasoning text when enabled.
        # if self.enable_reasoning:
        #     print("  - Extracting Reasoning texts...")
        #     df['text_reasoning'] = df.apply(lambda row: safe_get_run_data(row, key='assistant_reasoning'), axis=1)
        # else:
        #     df['text_reasoning'] = None # or empty string to avoid downstream errors

        # ==============================================================================
        # 1) Internal Confidence (probability)
        # ==============================================================================
        print("  - Calculating Internal Confidence (Probabilistic Scores)...")
        
        # a) Diagnosis probabilistic score (always computed)
        # JSON key: 'geometric_mean_probability'
        diag_stats = df['runs'].apply(
            lambda runs: internal.calculate_quantitative_features(runs, target_key='diagnosis_prob')
        )
        # Rename columns for clarity.
        df['ProbScore_Dx'] = diag_stats['Mean']
        df['ProbStability_Dx'] = diag_stats['Std']

        # Bottom-3 diagnosis probability focuses on the weakest diagnosis tokens
        # instead of the full-token geometric mean used by ProbScore_Dx.
        bottom3_stats = df['runs'].apply(
            lambda runs: internal.calculate_quantitative_features(runs, target_key='diagnosis_bottom3_prob')
        )
        df['Bottom3Prob_Dx'] = bottom3_stats['Mean']

        # b) Reasoning probabilistic score (optional)
        # JSON key: 'reason_prob'
        if self.enable_reasoning:
            # Reuse the same function with a different key.
            reas_stats = df['runs'].apply(
                lambda runs: internal.calculate_quantitative_features(runs, target_key='reason_prob')
            )
            # Rename columns.
            df['ProbScore_R'] = reas_stats['Mean']
            # Optional: store reasoning stability if needed.
            # df['Probabilistic_Stability_Reasoning'] = reas_stats['Std']


        # ==============================================================================
        # 2) Behavioral Confidence (consistency)
        # ==============================================================================
        print("  - Calculating Behavioral Consistency...")
        
        # A) Diagnosis consistency (always computed)
        # Use semantic embeddings for short text consistency.
        df['Consistency_Dx'] = df['runs'].apply(
            lambda runs: behavioral.calculate_consistency_score(
                [r.get('assistant_diagnosis', '') for r in runs], 
                self.config
            )
        )
        df['TraceConsistency_Dx'] = df['runs'].apply(
            lambda runs: behavioral.calculate_trace_consistency(
                [r.get('diagnostic_tool_trace', []) for r in runs]
            )
        )
        df['SemanticEntropy_Dx'] = df['runs'].apply(
            lambda runs: behavioral.calculate_semantic_entropy(
                [r.get('assistant_diagnosis', '') for r in runs],
                self.config
            )
        )
        # Keep the raw entropy for reporting, but expose a monotonic certainty
        # transform so higher values align with the rest of the confidence scores.
        df['SemanticCertainty_Dx'] = np.exp(-df['SemanticEntropy_Dx'])

        # B) Reasoning consistency (optional)
        # Reasoning is long text; embeddings are required for meaningful similarity.
        if self.enable_reasoning:
            print("    - Calculating Reasoning Consistency...")
            df['Consistency_R'] = df['runs'].apply(
                lambda runs: behavioral.calculate_consistency_score(
                    # Extract 'assistant_reasoning'.
                    [r.get('assistant_reasoning', '') for r in runs], 
                    self.config
                )
            )


        # # d) Self-reported confidence score
        # reported_features = df['runs'].apply(reported.calculate_reported_confidence_features)
        # df = pd.concat([df, reported_features], axis=1)
        # print('reported_features:', df['Reported_Confidence_Score'])

        # ==============================================================================
        # 3) Expressed Confidence (linguistic features)
        # ==============================================================================
        print("  - Calculating Expressed Confidence (Linguistic Metrics)...")
        
        enabled_linguistic_features = list(AVAILABLE_LINGUISTIC_CALCULATORS.keys())
        
        for feature_base_name in enabled_linguistic_features:
            calculator = AVAILABLE_LINGUISTIC_CALCULATORS.get(feature_base_name)
            
            if calculator:
                # --- Symmetric calculation logic ---
                
                # A) Diagnosis (suffix: _Dx)
                col_name_dx = f"{feature_base_name}_Dx"
                df[col_name_dx] = df.apply(
                    lambda row: _calculate_linguistic_avg(
                        row['runs'], 
                        key='assistant_diagnosis', 
                        calculator=calculator, 
                        ground_truth=None # no ground truth needed
                    ),
                    axis=1
                )
                print(f"    - Calculated Avg: {col_name_dx}")

                # B) Reasoning (suffix: _R), if enabled
                if self.enable_reasoning:
                    col_name_r = f"{feature_base_name}_R"
                    df[col_name_r] = df.apply(
                        lambda row: _calculate_linguistic_avg(
                            row['runs'], 
                            key='assistant_reasoning', 
                            calculator=calculator, 
                            ground_truth=None # no ground truth needed
                        ),
                        axis=1
                    )
                    print(f"    - Calculated Avg: {col_name_r}")
            else:
                print(f"    - WARNING: Calculator '{feature_base_name}' not found.")


        # ==============================================================================
        # 4) Final decision (majority vote)
        # ==============================================================================
        df['final_decision'] = df['runs'].apply(internal.majority_vote_decision)

        # ==============================================================================
        # 5) Final cleanup and output
        # ==============================================================================
        print("  - Finalizing DataFrame...")
        
        base_cols = ['hadm_id', 'ground_truth', 'dataset', 'final_decision']

        # Core score columns.
        score_cols = [
            'ProbScore_Dx',  
            'Bottom3Prob_Dx',
            'ProbStability_Dx',
            'Consistency_Dx',
            'TraceConsistency_Dx',
            'SemanticEntropy_Dx',
            'SemanticCertainty_Dx',
            'LingCert_Dx',
            'ConceptDensity_Dx',
            'ProbScore_R',
            'Consistency_R',
            'LingCert_R',
            'ConceptDensity_R',
        ]
        # Build final column list.
        final_cols_exist = base_cols + [c for c in score_cols if c in df.columns]        
        df_final = df[final_cols_exist].copy()
        
        # Preview output.
        print("\nFeature Engineering Output Preview:")
        print(df_final.head(2))

        print("\n--- Feature Engineering Complete ---")
        print(f"Successfully engineered features for {len(df_final)} records.")
        return df_final


def _calculate_linguistic_avg(runs, key, calculator, ground_truth=None, agg="mean"):
    """
    Iterate over runs, compute scores for the specified text key, and return the average.
    """
    if not runs or not isinstance(runs, list):
        return float("nan")
    
    scores = []
    for r in runs:
        # 1) Safely extract text.
        text = r.get(key, '')
        if not isinstance(text, str) or not text.strip():
            # Let calculators handle empty text if desired, but skipping is safer.
            continue
        
        # 2) Compute per-run score.
        # Calculators must handle empty strings and return 0.0 or NaN.
        try:
            score = calculator(text, ground_truth)
            score = float(score)
            # Filter NaNs (some calculators return NaN on empty text).
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue # skip failed runs

    if not scores:
        return float("nan")
        
    # 3) Return aggregate.
    return float(np.median(scores)) if agg=="median" else float(np.mean(scores))



def safe_get_run_data(row, run_index=3, key='assistant_diagnosis'):
    """
    Generic data accessor.
    Can extract 'assistant_diagnosis' or other text keys.
    """
    runs = row['runs']
    hadm_id = row.get('hadm_id', 'N/A')
    dataset = row.get('dataset', 'N/A') 

    try:
        # Core logic: try to access the requested run/key.
        if runs and len(runs) > run_index:
            return runs[run_index].get(key, '')
        else:
            # Not an error; data may be incomplete.
            print(f"  - Debug: Dataset={dataset}, HADM_ID={hadm_id} runs length ({len(runs)}) is insufficient for index {run_index}.")
            return ''
    except (IndexError, TypeError) as e:
        # --- Error encountered ---
        # Print useful diagnostics.
        print("\n" + "="*50)
        print("ERROR encountered while processing a row!")
        print(f"  - Dataset: {dataset}")
        print(f"  - HADM_ID: {hadm_id}")
        print(f"  - Error Type: {type(e).__name__}")
        print(f"  - Error Details: {e}")
        print(f"  - 'runs' column content for this row: {runs}")
        print("="*50 + "\n")
        # Return a safe value to allow .apply() to continue.
        return 'ERROR_OCCURRED'
