import random

# Define input and output file paths
input_file = "policy_outputs/EU_AI_Act_structured.txt"
output_file = "policy_outputs/selected_clause.txt"

# Read the structured policy document
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Extract policy clauses (excluding header)
policy_clauses = lines[2:]  # Skip the first two lines (header row and separator)

# Randomly select 10 unique policy clauses
random_clauses = random.sample(policy_clauses, 10)

# Write selected clauses to a new file
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(random_clauses)

print(f"10 random policy clauses have been saved to {output_file}")
