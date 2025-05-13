from collections import defaultdict

def load_policy_by_section(file_path):
    grouped_policies = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip the first two header lines
    for line in lines[2:]:
        line = line.strip()
        if line.startswith('|') and '|' in line[1:]:
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 3:
                section_num = parts[0]
                grouped_policies[section_num].append(line)

    return dict(grouped_policies)

# Example usage
if __name__ == "__main__":
    file_path = "policies/CCPA.txt"  
    policy_dict = load_policy_by_section(file_path)


    print(policy_dict)
