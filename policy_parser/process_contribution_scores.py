import os
import pandas as pd

# Define the folder containing the evaluation files
input_folder = "outputs"

# Initialize storage for contribution scores
contrib_data = {}

# Helper function to map file name to short policy-model label
def get_label(filename):
    parts = filename.replace("evaluation_", "").replace(".txt", "").split("_")
    policy = parts[0]
    model = parts[1][0]  # First letter of the model name
    return f"{policy}_{model}"

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        label = get_label(filename)
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
            in_table = False
            for line in lines:
                if "| Model Card Section | Compliance Contribution Score (0-5) | Reasoning |" in line:
                    in_table = True
                    continue
                if in_table:
                    if line.strip().startswith("|") and not line.strip().startswith("|---"):
                        parts = [part.strip() for part in line.strip().split("|")[1:-1]]
                        if len(parts) >= 2:
                            section, score = parts[0], parts[1]
                            try:
                                score = int(score)
                                if section not in contrib_data:
                                    contrib_data[section] = {}
                                contrib_data[section][label] = score
                            except ValueError:
                                continue
                    elif not line.strip().startswith("|"):
                        break

# Convert to DataFrame
df = pd.DataFrame.from_dict(contrib_data, orient='index').sort_index()
df.index.name = "Section"

# Save to Markdown table in txt file
with open("contribution_matrix_summary.txt", "w", encoding="utf-8") as f:
    f.write(df.reset_index().to_markdown(index=False))

print("Saved contribution matrix to 'contribution_matrix_summary.txt'")
