#!/usr/bin/env python3
"""
Script to parse relevancy_rating.txt file and append to existing Excel files for GDPR and Colorado policies.
Extracts: card (set to "mc2"), Section, Article, and score (Compliance Contribution Score).
"""

import re
import pandas as pd
from pathlib import Path

def parse_relevancy_rating(file_path):
    """
    Parse the relevancy_rating.txt file and extract the required information.
    
    Args:
        file_path (str): Path to the relevancy_rating.txt file
        
    Returns:
        dict: Dictionary with 'GDPR' and 'Colorado' keys containing parsed data
    """
    gdpr_data = []
    colorado_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content by policy sections (separated by =)
    policy_sections = re.split(r'={80,}', content.strip())
    
    print(f"Found {len(policy_sections)} policy sections")
    
    for i, section in enumerate(policy_sections):
        if not section.strip():
            continue
            
        # Extract policy name
        policy_match = re.search(r'Policy:\s*(.+)\.txt', section)
        if not policy_match:
            continue
            
        policy_name = policy_match.group(1).strip()
        print(f"Processing section {i+1}: Policy = '{policy_name}'")
        
        # Extract articles range
        articles_match = re.search(r'Articles:\s*(.+)', section)
        if not articles_match:
            continue
            
        articles_text = articles_match.group(1).strip()
        
        # Find all section headers
        section_headers = re.findall(r'-\s*Section:\s*(.+?)(?=\n|$)', section)
        print(f"  Found {len(section_headers)} section headers: {section_headers[:3]}...")
        
        # Process each section header
        for section_header in section_headers:
            section_name = section_header.strip()
            
            # Find the table that follows this section header
            # Look for the section header and then find the next table
            section_start = section.find(f"- Section: {section_name}")
            if section_start == -1:
                continue
                
            # Find the next table after this section
            remaining_text = section[section_start:]
            table_match = re.search(r'\|.*\n\|-+\|.*\n(\|.*\n)*', remaining_text)
            
            if table_match:
                table = table_match.group(0)
                
                # Parse table rows
                rows = table.strip().split('\n')
                for row in rows:
                    if '|' not in row or 'Article No.' in row or '---' in row:
                        continue
                        
                    # Split row by | and clean up
                    cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                    if len(cells) >= 4:
                        article_no = cells[0]
                        # Updated regex to handle Colorado article numbers like 6-1-1701
                        if not re.match(r'^[\d-]+$', article_no):
                            continue
                            
                        score = cells[3]  # Compliance Contribution Score is the 4th column
                        
                        # Add to appropriate dataset based on policy
                        data_entry = {
                            'card': 'mc6',
                            'Section': section_name,
                            'Article': article_no,
                            'score': score
                        }
                        
                        if policy_name == 'GDPR':
                            gdpr_data.append(data_entry)
                        elif policy_name == 'Colorado':
                            colorado_data.append(data_entry)
    
    print(f"Final counts - GDPR: {len(gdpr_data)}, Colorado: {len(colorado_data)}")
    
    return {
        'GDPR': gdpr_data,
        'Colorado': colorado_data
    }

def append_to_excel(data_dict, output_base):
    """
    Append the parsed data to existing Excel files for each policy.
    If files don't exist, create them.
    
    Args:
        data_dict (dict): Dictionary with 'GDPR' and 'Colorado' keys containing parsed data
        output_base (str): Base path for the output Excel files
    """
    for policy, data in data_dict.items():
        if data:
            df_new = pd.DataFrame(data)
            excel_path = f"{output_base}_{policy}.xlsx"
            
            # Check if file exists
            if Path(excel_path).exists():
                # Read existing file and append new data
                try:
                    df_existing = pd.read_excel(excel_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_excel(excel_path, index=False, engine='openpyxl')
                    print(f"{policy} data appended to existing Excel file: {excel_path}")
                    print(f"  - Existing rows: {len(df_existing)}")
                    print(f"  - New rows added: {len(df_new)}")
                    print(f"  - Total rows: {len(df_combined)}")
                except Exception as e:
                    print(f"Error reading existing file {excel_path}: {e}")
                    # If reading fails, create new file
                    df_new.to_excel(excel_path, index=False, engine='openpyxl')
                    print(f"{policy} data saved to new Excel file: {excel_path}")
                    print(f"  - Total rows: {len(df_new)}")
            else:
                # Create new file if it doesn't exist
                df_new.to_excel(excel_path, index=False, engine='openpyxl')
                print(f"{policy} data saved to new Excel file: {excel_path}")
                print(f"  - Total rows: {len(df_new)}")
            
            print(f"  - Unique sections: {df_new['Section'].nunique()}")
            print(f"  - Unique articles: {df_new['Article'].nunique()}")
        else:
            print(f"No data found for {policy}")

def main():
    input_file = 'relevancy_rating.txt'
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Parse the file
    print(f"Parsing {input_file}...")
    data_dict = parse_relevancy_rating(input_file)
    
    if not data_dict['GDPR'] and not data_dict['Colorado']:
        print("No data found to parse.")
        return
    
    # Append to existing Excel files or create new ones
    output_base = 'relevancy_rating_parsed'
    append_to_excel(data_dict, output_base)
    
    # Display sample of parsed data
    print("\nSample of parsed data:")
    for policy, data in data_dict.items():
        if data:
            print(f"\n{policy} - Sample data:")
            df = pd.DataFrame(data)
            print(df.head(5))

if __name__ == "__main__":
    main() 