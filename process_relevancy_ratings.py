import pandas as pd

def process_relevancy_ratings(input_file, output_file):
    """Group by Section/Article and keep each group's max score."""
    try:
        print(f"Reading {input_file}...")
        df = pd.read_excel(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        required_columns = ['Section', 'Article', 'score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print("\nFirst few rows of original data:")
        print(df.head())
        
        print("\nGrouping by Section and Article...")
        grouped = df.groupby(['Section', 'Article'])['score'].max().reset_index()
        grouped = grouped.rename(columns={'score': 'max_score'})
        
        print(f"Grouped data shape: {grouped.shape}")
        print("\nFirst few rows of grouped data:")
        print(grouped.head())
        
        print(f"\nSaving results to {output_file}...")
        grouped.to_excel(output_file, index=False)
        
        print(f"Successfully processed {input_file}")
        print(f"Results saved to {output_file}")
        
        print(f"\nSummary:")
        print(f"Original rows: {len(df)}")
        print(f"Unique Section-Article combinations: {len(grouped)}")
        print(f"Score range: {grouped['max_score'].min()} to {grouped['max_score'].max()}")
        
        return grouped
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

def main():
    """Main function to process both files."""
    gdpr_input = "relevancy_rating_parsed_GDPR.xlsx"
    colorado_input = "relevancy_rating_parsed_Colorado.xlsx"
    
    gdpr_output = "relevancy_rating_parsed_GDPR_summarized.xlsx"
    colorado_output = "relevancy_rating_parsed_Colorado_summarized.xlsx"
    
    print("=" * 60)
    print("PROCESSING RELEVANCY RATING FILES")
    print("=" * 60)
    
    print("\n" + "=" * 40)
    print("PROCESSING GDPR FILE")
    print("=" * 40)
    gdpr_results = process_relevancy_ratings(gdpr_input, gdpr_output)
    
    print("\n" + "=" * 40)
    print("PROCESSING COLORADO FILE")
    print("=" * 40)
    colorado_results = process_relevancy_ratings(colorado_input, colorado_output)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    if gdpr_results is not None:
        print(f"✓ GDPR file processed successfully")
        print(f"  Output: {gdpr_output}")
    
    if colorado_results is not None:
        print(f"✓ Colorado file processed successfully")
        print(f"  Output: {colorado_output}")
    
    if gdpr_results is not None and colorado_results is not None:
        print(f"\nFinal Summary:")
        print(f"GDPR: {len(gdpr_results)} unique Section-Article combinations")
        print(f"Colorado: {len(colorado_results)} unique Section-Article combinations")

if __name__ == "__main__":
    main() 