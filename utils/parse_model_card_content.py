import os
from dotenv import load_dotenv

def parse_model_card_content(model_card_content):
    """Parse model card content into sections"""
    sections_content = {}
    current_section = None
    current_content = []
    
    for line in model_card_content.split('\n'):
        if line.startswith('|') and '|' in line[1:]:
            # This is a table row
            if current_section:
                current_content.append(line)
        elif line.strip():
            # This is a section header
            if current_section and current_content:
                sections_content[current_section] = '\n'.join(current_content)
            current_section = line.strip()
            current_content = []
    
    # Add the last section
    if current_section and current_content:
        sections_content[current_section] = '\n'.join(current_content)
    print(f"sections_content is {sections_content}")
    return sections_content