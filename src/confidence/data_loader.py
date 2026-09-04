import os
import pandas as pd

class DataLoader:
    """
    Load and merge aggregated datasets using paths from the config.
    """
    def __init__(self, config: dict):
        """
        Initialize the DataLoader.

        Args:
            config (dict): Full configuration loaded from YAML.
        """
        self.config = config
        datasets_cfg = config['datasets']
        benchmark = config.get('benchmark')
        self.datasets_to_load = datasets_cfg.get(benchmark, [])
        self.aggregated_dir = config['data']['aggregated_dir']
        print("DataLoader initialized.")
        print(f"  - Target datasets: {self.datasets_to_load}")
        print(f"  - Aggregated data source: {self.aggregated_dir}")

    def load_and_combine_data(self) -> pd.DataFrame | None:
        """
        Load aggregated JSONL files for all datasets and combine into a master DataFrame.
        
        Returns:
            pd.DataFrame | None: Combined DataFrame, or None if nothing was loaded.
        """
        all_dfs = []
        print("\n--- Starting Data Loading and Combination ---")

        for dataset in self.datasets_to_load:
            file_path = os.path.join(self.aggregated_dir, f"{dataset}_multi_run.jsonl")
            
            print(f"  - Attempting to load: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"    - WARNING: File not found, skipping dataset '{dataset}'.")
                continue

            try:
                df = pd.read_json(file_path, lines=True)
                
                # Add a column to mark the dataset source for downstream analysis.
                df['dataset'] = dataset
                
                all_dfs.append(df)
                print(f"    - SUCCESS: Loaded {len(df)} records for '{dataset}'.")

            except Exception as e:
                print(f"    - ERROR: Failed to load or process file for '{dataset}'. Reason: {e}")

        if not all_dfs:
            print("\n❌ CRITICAL: No data could be loaded. The pipeline cannot continue.")
            return None

        # Merge all individual DataFrames into a single master DataFrame.
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        print("\n--- Data Loading Complete ---")
        print(f"✅ Successfully combined data for {len(all_dfs)} datasets.")
        print(f"Total records in Master DataFrame: {len(master_df)}")
        
        return master_df
