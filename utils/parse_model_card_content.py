from collections import defaultdict

def parse_model_card_by_section(file_path):
    sections = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip the markdown title (if any), and header rows
    for line in lines[2:]:  # Skip: 0 = title, 1 = column header, 2 = separator
        line = line.strip()
        if line.startswith('|') and '|' in line[1:]:
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 2:
                section, content = parts[0], parts[1]
                sections[section] = content

    return sections

# Example usage
if __name__ == "__main__":
    file_path = "model_cards/ai_resume_screening_tool_model_card.txt"
    sections_content = parse_model_card_by_section(file_path)

    print(sections_content)