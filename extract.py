import os
import re
import pandas as pd

def extract_scores_from_folder(folder_path, output_excel_path):
    policy_pattern = re.compile(r'Policy:\s*([^.]+)\.txt')
    mc_pattern = re.compile(r'(mc\d+)_relevancy_rating\.txt')

    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith("_relevancy_rating.txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            # Extract MC ID from filename (e.g., "mc3")
            mc_match = mc_pattern.match(filename)
            mc_id = mc_match.group(1) if mc_match else "unknown_mc"

            # Split by blocks using "Policy:" markers
            policy_blocks = content.split("Policy:")
            for block in policy_blocks[1:]:
                lines = block.strip().splitlines()

                # Extract policy prefix
                policy_line = lines[0]
                policy_prefix = policy_line.strip().replace(".txt", "")

                section_blocks = re.split(r'- Section: ', block)[1:]
                for section_block in section_blocks:
                    lines = section_block.strip().splitlines()
                    if not lines:
                        continue
                    section_name = lines[0].strip()

                    # Extract table lines only
                    for i, line in enumerate(lines):
                        if "Compliance Contribution Score" in line:
                            table_lines = lines[i+1:]
                            break
                    else:
                        continue  # no table found

                    for row in table_lines:
                        if '|' not in row or '---' in row:
                            continue
                        cols = [col.strip() for col in row.strip('|').split('|')]
                        if len(cols) < 4:
                            continue
                        try:
                            article_number = cols[0]
                            score = int(cols[3])
                            full_article = f"{policy_prefix}.{article_number}"
                            all_data.append([mc_id, section_name, full_article, score])
                        except ValueError:
                            continue

    df = pd.DataFrame(all_data, columns=["MC Source", "Section", "Article", "Compliance Contribution Score"])
    df.to_excel(output_excel_path, index=False)
    print(f"✅ Exported clean file to: {output_excel_path}")

# Run the extraction
extract_scores_from_folder("relevancy_ratings", "final_clean_relevancy_scores.xlsx")
