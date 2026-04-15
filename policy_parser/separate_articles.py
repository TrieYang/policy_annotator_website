import os
import pandas as pd

# Define input file and output directory
input_file = "policy_outputs/EU_AI_Act_structured.txt"
output_dir = "policy_outputs/Articles_structured"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the structured text file into a DataFrame
df = pd.read_csv(input_file, delimiter="|", skipinitialspace=True, dtype=str)

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Group by article number and save each article to a separate file
for article_number, group in df.groupby("Article Number"):
    article_filename = os.path.join(output_dir, f"{article_number.strip().replace(' ', '_').rstrip('_')}.txt")

    
    # Format content to include clause numbers
    article_content = "\n".join(group.apply(lambda row: f"| {row['Article Number']} | {row['Clause Number']} | {row['Content']} |", axis=1))
    
    # Save to file
    with open(article_filename, "w", encoding="utf-8") as file:
        file.write(article_content)

print(f"Articles saved in: {output_dir}")


