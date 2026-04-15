import random

# Define input and output file paths
input_file = "model_card_outputs/retail_customer_analytics_ai_model_card.txt"
output_file = "model_card_outputs/selected_clause.txt"

# Read the structured policy document
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Extract policy clauses (excluding header)
mc_clauses = lines[2:]  # Skip the first two lines (header row and separator)

# Randomly select 10 unique policy clauses
random_sections = random.sample(mc_clauses, 10)

# Write selected clauses to a new file
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(random_sections)

print(f"10 random policy clauses have been saved to {output_file}")