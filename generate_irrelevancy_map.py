import os
import json
from collections import defaultdict, Counter

folder_path = "irrelevant_summary"  
file_names = [f"mc{i}_summary.txt" for i in range(1, 9)]
threshold = 0.75  # e.g., appears in 6/8 files
min_votes = int(threshold * len(file_names))

# policy -> section -> article -> count
article_counts = defaultdict(lambda: defaultdict(Counter))

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for policy, sections in data.items():
            for section, articles in sections.items():
                for article in articles:
                    article_counts[policy][section][article] += 1

universal_irrelevancy_map = defaultdict(lambda: defaultdict(list))

for policy, sections in article_counts.items():
    for section, article_counter in sections.items():
        for article, count in article_counter.items():
            if count >= min_votes:
                universal_irrelevancy_map[policy][section].append(article)

import re

def extract_numeric_key(article_name):
    # Extract all numeric parts from the article name
    return [int(part) if part.isdigit() else part for part in re.split(r'[^\d]+', article_name) if part]

# Sort article lists by numeric value
for policy in universal_irrelevancy_map:
    for section in universal_irrelevancy_map[policy]:
        universal_irrelevancy_map[policy][section].sort(key=extract_numeric_key)

output_file = "universal_irrelevancy_map.json"
with open(output_file, "w", encoding="utf-8") as out:
    json.dump(universal_irrelevancy_map, out, indent=2)

print(f"✅ Universal irrelevancy map saved to {output_file}")
