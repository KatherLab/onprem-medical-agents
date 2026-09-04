# main_analysis.py
import yaml
import argparse
import os
import pandas as pd
from pathlib import Path


from confidence.data_loader import DataLoader
from confidence.feature_engineering import FeatureEngineer
from confidence.analysis import Analyzer

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "analysis_config.yaml"


def main():
    """
    Main function that runs the complete confidence-analysis pipeline in sequence.
    """
    # --- 1. Configure and load settings ---
    print("="*60)
    print("🚀 STARTING CONFIDENCE ANALYSIS PIPELINE 🚀")
    print("="*60)

    # Use argparse to allow a configuration file to be specified on the command line
    parser = argparse.ArgumentParser(description="Run the full confidence analysis pipeline.")
    parser.add_argument(
        '--config', 
        default=str(DEFAULT_CONFIG_PATH),
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()
    
    config_path = args.config
    print(f"🔩 Loading configuration from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Configuration file not found at '{config_path}'.")
        print("Pipeline terminated.")
        return # Use return instead of exit() for cleaner control flow

    # --- 2. Run pipeline steps ---

    # a. First, construct the expected final feature-file path
    processed_output_path = os.path.join(
        config.get('data', {}).get('processed_dir', ''),
        'master_feature_dataframe.csv'
    )
    
    df_featured = None # Initialize the variable

    # b. Check whether the file exists
    if os.path.exists(processed_output_path):
        print(f"\n✅ Found existing featured DataFrame at: {processed_output_path}")
        print("    --> Skipping data loading and feature engineering steps.")
        try:
            # Load the file directly when it exists
            df_featured = pd.read_csv(processed_output_path)
            print(f"    --> Successfully loaded {len(df_featured)} records.")
        except Exception as e:
            print(f"    ⚠️ WARNING: Could not load the existing file. Error: {e}")
            print("    --> Proceeding with the full pipeline.")
            df_featured = None # Reset the variable after a load failure so it can be regenerated

    # c. Run the full pipeline only when the file is absent or loading fails
    if df_featured is None:
        print("\n--- Running full data processing pipeline ---")
    
        # Step 1: data loading
        print("\n[STEP 1/3] Loading and combining data...")
        data_loader = DataLoader(config)
        df_master = data_loader.load_and_combine_data()

        # Check whether the previous step succeeded
        if df_master is None:
            print("\n❌ PIPELINE HALTED: Data loading failed. Please check the logs above.")
            return

        # Step 2: feature engineering
        print("\n[STEP 2/3] Engineering features...")
        feature_engineer = FeatureEngineer(config)
        df_featured = feature_engineer.run(df_master)

        # Check whether the previous step succeeded
        if df_featured is None or df_featured.empty:
            print("\n❌ PIPELINE HALTED: Feature engineering failed or produced no data.")
            return

        # (Optional) Save the intermediate feature DataFrame.
        processed_output_path = os.path.join(
            config.get('data', {}).get('processed_dir', ''),
            'master_feature_dataframe.csv'
        )
        os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
        df_featured.to_csv(processed_output_path, index=False)
        print(f"\n💾 Intermediate featured DataFrame saved to: {processed_output_path}")

    # Step 3: analysis and visualization
    print("\n[STEP 3/3] Performing analysis and generating visualizations...")
    analyzer = Analyzer(config)
    analyzer.run(df_featured)

    # --- 4. Pipeline complete ---
    print("\n" + "="*60)
    print("✅ PIPELINE FINISHED SUCCESSFULLY ✅")
    print("="*60)


if __name__ == "__main__":
    main()