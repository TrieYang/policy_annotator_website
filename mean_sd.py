import os
import re
import numpy as np
from collections import defaultdict
from openpyxl import Workbook

# Folder containing mc*_relevancy_rating.txt files
folder_path = "relevancy_ratings"

# Data structure to store scores: scores[(section, policy.article)] = list of scores
scores = defaultdict(list)

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith("mc") and filename.endswith("_relevancy_rating.txt"):
        filepath = os.path.join(folder_path, filename)

        current_section = None
        current_policy = None
        in_table = False

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Detect policy name (e.g., "Policy: EU.txt")
                if line.startswith("Policy:"):
                    current_policy = line.split(":", 1)[1].strip().replace(".txt", "").strip()
                    in_table = False

                # Detect section name
                elif line.startswith("- Section:"):
                    current_section = line.split(":", 1)[1].strip()
                    in_table = False

                # Detect the start of a relevant markdown table
                elif line.startswith("| Article No.") and "Compliance Contribution Score" in line:
                    in_table = True

                # Parse data rows
                elif in_table and re.match(r"^\|\s*[\w.]+", line):
                    parts = [p.strip() for p in line.strip("|").split("|")]
                    if len(parts) >= 4:
                        article_no = parts[0]
                        score_str = parts[3]  # Compliance Contribution Score
                        try:
                            score = float(score_str)
                            full_article_id = f"{current_policy}.{article_no}"
                            scores[(current_section, full_article_id)].append(score)
                        except ValueError:
                            continue

# Create Excel workbook and sheet
wb = Workbook()
ws = wb.active
ws.title = "Compliance Summary"

# Write header
ws.append(["Section", "Article", "Mean", "StdDev"])

# Write data
for key in sorted(scores):
    section, full_article = key
    values = scores[key]
    mean = np.mean(values)
    std = np.std(values)
    ws.append([section, full_article, round(mean, 2), round(std, 2)])

# Save to Excel file
output_excel = "section_article_scores_summary.xlsx"
wb.save(output_excel)
print(f"✅ Results saved to Excel: {output_excel}")
