import pandas as pd
import numpy as np

def analyze_dataset_ranges():
    """Analyze the ranges of all features in the BIDMC dataset"""
    
    # Load the dataset
    data = pd.read_csv('merged_bidmc_1s.csv')
    
    # Print column names to see what we're working with
    print("\nAvailable columns in dataset:")
    print(data.columns.tolist())
    
    # Get basic statistics for each column
    stats = data.describe()
    
    # Print full analysis for all columns except Time
    print("\nComplete Dataset Analysis:")
    print("-------------------------")
    analysis = pd.DataFrame({
        'Min': stats.loc['min'],
        'Max': stats.loc['max'],
        'Mean': stats.loc['mean'],
        'Std': stats.loc['std'],
        '25th Percentile': stats.loc['25%'],
        '75th Percentile': stats.loc['75%']
    })
    
    # Print analysis for all columns except Time
    for column in data.columns:
        if column.lower() != 'time':
            print(f"\n{column}:")
            print(analysis.loc[column].round(3))

if __name__ == "__main__":
    analyze_dataset_ranges() 