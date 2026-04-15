import os
import pandas as pd

# Define the folder containing the evaluation files
input_folder = "outputs"
output_folder = "policy_tables"
os.makedirs(output_folder, exist_ok=True)

# Initialize storage for policy-wise data
policy_data = {}

# Helper to extract model and policy names
def get_policy_model_label(filename):
    name = filename.replace("evaluation_", "").replace(".txt", "")
    policy, model = name.split("_")
    return policy, model

# Process each file
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        policy, model = get_policy_model_label(filename)
        label = f"{model}"
        path = os.path.join(input_folder, filename)

        with open(path, "r", encoding="utf-8") as f:
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
                                if policy not in policy_data:
                                    policy_data[policy] = {}
                                if section not in policy_data[policy]:
                                    policy_data[policy][section] = {}
                                policy_data[policy][section][label] = score
                            except ValueError:
                                continue
                    elif not line.strip().startswith("|"):
                        break

# Generate one markdown table per policy
for policy, section_scores in policy_data.items():
    df = pd.DataFrame.from_dict(section_scores, orient="index").sort_index()
    df.index.name = "Section"
    df["Mean"] = df.mean(axis=1)
    df["Standard Deviation"] = df.std(axis=1)
    md_table = df.reset_index().to_markdown(index=False)
    with open(os.path.join(output_folder, f"{policy}_contribution_matrix.txt"), "w", encoding="utf-8") as f:
        f.write(md_table)

print("All policy-specific markdown tables saved in 'policy_tables/'")
