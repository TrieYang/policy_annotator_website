#!/usr/bin/env python3
"""
Script to filter Excel file by max_score column values (0 or 1)
"""

import pandas as pd
import os
from datetime import datetime

def filter_excel_by_max_score(input_file, output_file=None):
    """
    Filter Excel file to keep only rows where max_score is NOT 0 or 1
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file (optional)
    
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
        
        # Filter rows where max_score is NOT 0 or 1 (exclude rows with max_score = 0 or max_score = 1)
        filtered_df = df[~df['max_score'].isin([0, 1])]
        
        print(f"Filtered data shape: {filtered_df.shape}")
        print(f"Rows excluded (max_score = 0): {len(df[df['max_score'] == 0])}")
        print(f"Rows excluded (max_score = 1): {len(df[df['max_score'] == 1])}")
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_filtered_max_score_not_0_1_{timestamp}.xlsx"
        
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
    """Main function to run the script"""
    # Input file path
    input_file = "relevancy_rating_parsed_Colorado_summarized.xlsx"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found in current directory")
        print("Please make sure the file exists or update the input_file variable")
        return
    
    # Filter the Excel file
    output_file = filter_excel_by_max_score(input_file)
    
    if output_file:
        print(f"\nSuccess! Filtered data saved to: {output_file}")
    else:
        print("Failed to process the Excel file")

if __name__ == "__main__":
    main()
