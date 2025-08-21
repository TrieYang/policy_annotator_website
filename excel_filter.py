#!/usr/bin/env python3
"""
Flexible Excel filtering script that can filter by max_score column values
"""

import pandas as pd
import os
import argparse
from datetime import datetime

def filter_excel_by_max_score(input_file, output_file=None, score_values=[0, 1]):
    """
    Filter Excel file to keep only rows where max_score does NOT match specified values
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file (optional)
        score_values (list): List of score values to exclude (default: [0, 1])
    
    Returns:
        str: Path to the output file
    """
    try:
        # Read the Excel file
        print(f"Reading Excel file: {input_file}")
        df = pd.read_excel(input_file)
        
        # Display basic info about the data
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if max_score column exists
        if 'max_score' not in df.columns:
            print("Error: 'max_score' column not found in the Excel file")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Display unique values in max_score column
        unique_scores = df['max_score'].unique()
        print(f"Unique values in max_score column: {sorted(unique_scores)}")
        
        # Filter rows where max_score does NOT match specified values (exclude rows with specified scores)
        filtered_df = df[~df['max_score'].isin(score_values)]
        
        print(f"Filtered data shape: {filtered_df.shape}")
        
        # Show count for excluded score values
        for score in score_values:
            count = len(df[df['max_score'] == score])
            print(f"Rows excluded (max_score = {score}): {count}")
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(input_file)[0]
            score_str = "_".join(map(str, score_values))
            output_file = f"{base_name}_filtered_max_score_not_{score_str}_{timestamp}.xlsx"
        
        # Save filtered data to new Excel file
        filtered_df.to_excel(output_file, index=False)
        print(f"Filtered data saved to: {output_file}")
        
        return output_file
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def main():
    """Main function to run the script from command line"""
    parser = argparse.ArgumentParser(description='Filter Excel file by max_score column values')
    parser.add_argument('input_file', help='Path to input Excel file')
    parser.add_argument('-o', '--output', help='Path to output Excel file (optional)')
    parser.add_argument('-s', '--scores', nargs='+', type=int, default=[0, 1], 
                       help='Score values to exclude (default: 0 1)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file '{args.input_file}' not found")
        return
    
    # Filter the Excel file
    output_file = filter_excel_by_max_score(args.input_file, args.output, args.scores)
    
    if output_file:
        print(f"\nSuccess! Filtered data saved to: {output_file}")
    else:
        print("Failed to process the Excel file")

if __name__ == "__main__":
    main()
