import pandas as pd
import json
import os
import argparse
from typing import Dict, List

def load_results(results_dir: str) -> pd.DataFrame:
    all_results = []
    
    # Walk through all experiment directories
    for exp_type in ['batch_size', 'learning_rate', 'weight_decay', 'stopping_criteria']:
        exp_dirs = [d for d in os.listdir(results_dir) if d.startswith(exp_type)]
        
        for exp_dir in exp_dirs:
            # Load results table
            results_path = os.path.join(results_dir, exp_dir, "results_table.csv")
            if os.path.exists(results_path):
                df = pd.read_csv(results_path, index_col=[0,1])
                
                # Add experiment info
                param_type = exp_dir.split('_')[0]
                param_value = '_'.join(exp_dir.split('_')[1:])
                df['experiment_type'] = param_type
                df['parameter_value'] = param_value
                
                all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, axis=0)
    
    return combined_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load and combine all results
    combined_results = load_results(args.results_dir)
    
    # Save to Excel with multiple sheets
    output_path = os.path.join(args.results_dir, "all_results.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        # Write raw data
        combined_results.to_excel(writer, sheet_name='Raw Data')
        
        # Create summary sheets for each experiment type
        for exp_type in combined_results['experiment_type'].unique():
            df_exp = combined_results[combined_results['experiment_type'] == exp_type]
            df_exp.to_excel(writer, sheet_name=f'{exp_type}_summary')
    
    print(f"Saved combined results to: {output_path}")

if __name__ == "__main__":
    main() 