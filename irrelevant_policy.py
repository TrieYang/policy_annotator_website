import re
import json
from collections import defaultdict

def parse_policy_scores_to_zero(filepath):
    # Create a nested defaultdict structure: policy -> section -> articles
    zero_score_articles = defaultdict(lambda: defaultdict(list))
    current_section = None
    current_policy = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect policy name
        if line.startswith("Policy:"):
            current_policy = line.split(":")[1].strip().replace(".txt", "")

        # Detect section name (support both '- Section:' and 'Section:')
        elif line.startswith("- Section:") or line.startswith("Section:"):
            current_section = line.split(":", 1)[1].strip()

        # Detect table row
        elif re.match(r'^\|', line) and not line.startswith("| Article No."):
            cols = [col.strip() for col in line.strip('|').split('|')]

            if len(cols) >= 4:
                article_no = cols[0]
                contribution_score = cols[3]

                # Check for contribution score of 0
                if contribution_score == '0':
                    article_id = f"{current_policy}.Art.{article_no}"
                    zero_score_articles[current_policy][current_section].append(article_id)

    # Convert defaultdict to regular dict for JSON serialization
    result = {policy: dict(sections) for policy, sections in zero_score_articles.items()}
    return result


file_path = "mc8_relevancy_rating.txt"  # <- your input file
zero_score_summary = parse_policy_scores_to_zero(file_path)

# Save to txt file
output_path = "mc8_summary.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(zero_score_summary, f, indent=2)

