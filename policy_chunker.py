import json

def get_chunking_prompt(sections=None, irre=True):
    if sections is None:
        sections = [
            "System Name", "Versioning Information", "Primary Developer/Org",
            "Contact Info", "System Overview", "Primary intended uses",
            "Primary intended users", "Out-of-scope use cases", "Terms and conditions",
            "Current legal compliance status", "Dataset Description", "Collection Method",
            "Bias Mitigation Measures", "Usage Constraints", "Summary of Performance Assessment",
            "Disaggregated Performance", "Testing Contexts",
            "Evaluations for Edge Cases or Adversarial Inputs", "Potential Risks and Harms",
            "Actions taken", "Misuse Scenarios", "Human Oversight", "Update Frequency"
        ]
    
    sections_str = json.dumps(sections)
    if irre:
        return f"""Analyze the following markdown table policy document and split it into chunks for the specified model card sections.

        Return ONLY a JSON object where each key is a model card section and the value is an array of chunk boundaries in the format:
        {{
        "System Name": [
            {{ "s": "Article X", "e": "Article Y" }},
            ...
        ],
        "Contact Info": [
            {{ "s": "Article X", "e": "Article Y" }},
            ...
        ],
        ...
        }}

        **s means start, e means end**

        Definitions and Rules:

        1. An **Article** is defined as **all rows that share the same first column value** (e.g., "Article 3").  
        - If "Article 3" appears in multiple rows, it still counts as **one Article**.
        - You must process articles by their article numbers, not by rows.

        2. Each chunk must contain between **5 and 10 Articles** (not rows), **unless articles are isolated due to irrelevance filtering**, in which case **smaller chunks or single-article chunks are acceptable**.

        3. **CRITICAL: Chunk boundaries must not span irrelevant articles.**
        - If only Article 7 and Article 11 are relevant, then valid chunks would be:
            * `{{"s": "Article 7", "e": "Article 7"}}`
            * `{{"s": "Article 11", "e": "Article 11"}}`
        - NOT `{{"s": "Article 7", "e": "Article 11"}}`, because that wrongly includes Articles 8–10.

        4. Keep adjacent relevant articles grouped where possible — as long as no irrelevant articles are in between.

        5. Only include articles that are **not listed as irrelevant** for that section. Irrelevant articles **must be excluded entirely**.

        6. If a section has no relevant articles, return an empty array: `[]`.

        7. Only process the following sections: {sections_str}

        Input:
        - `Policy Document:` {{POLICY_DOC}}
        - `Section-Specific Irrelevant Article Lists:` {{IRRE_LIST}}

        Output:
        - **Return ONLY the JSON object** as described. **Do NOT include any explanation or extra text.**
        """
    else:
        return f"""Analyze the following markdown table policy document and split it into evaluation chunks.

            Return ONLY a JSON array of chunk boundaries in the format:
            [
            {{ 'start': "Article X", "end": "Article Y" }},
            ...
            ]

            Rules:
            1. An "Article" is defined as **all rows that share the same first column value** (e.g., "Article 3"). 
            - **Do not** treat each row as a separate article. For example, if "Article 3" appears in 68 rows, it still counts as **one single Article**.
            2. Each chunk must contain between 5 to 10 such Articles depending on lengths of articles(not rows).
            3. Keep related Articles together if possible.

            Return only the JSON array. 
            **Do not include any extra text than a single JSON array**.

            Policy Document:
            {{POLICY_DOC}}"""

def parse_chunk_response(response, irre=True):
    """Parse Claude's chunking response into a dictionary of section-specific chunk boundaries."""
    if irre:
        try:
            chunks = json.loads(response)
            # Convert each section's chunks into a list of (start, end) tuples
            return {
                section: [(chunk["s"], chunk["e"]) for chunk in section_chunks]
                for section, section_chunks in chunks.items()
            }
        except Exception as e:
            print(f"Error parsing chunk response: {e}")
            # Fallback to evaluating everything at once for all sections
            return {
                section: [(1, 999)]  # Large end number to include all articles
                for section in [
                    "System Name", "Versioning Information", "Primary Developer/Org",
                    "Contact Info", "System Overview", "Primary intended uses",
                    "Primary intended users", "Out-of-scope use cases", "Terms and conditions",
                    "Current legal compliance status", "Dataset Description", "Collection Method",
                    "Bias Mitigation Measures", "Usage Constraints", "Summary of Performance Assessment",
                    "Disaggregated Performance", "Testing Contexts",
                    "Evaluations for Edge Cases or Adversarial Inputs", "Potential Risks and Harms",
                    "Actions taken", "Misuse Scenarios", "Human Oversight", "Update Frequency"
                ]
            }
    else:
        try:
            chunks = json.loads(response)
            return [(chunk["start"], chunk["end"]) for chunk in chunks]
        except Exception as e:
            print(f"Error parsing chunk response: {e}")

def get_section_groups():
    """Split sections into five groups for processing."""
    all_sections = [
        "System Name", "Versioning Information", "Primary Developer/Org",
        "Contact Info", "System Overview", "Primary intended uses",
        "Primary intended users", "Out-of-scope use cases", "Terms and conditions",
        "Current legal compliance status", "Dataset Description", "Collection Method",
        "Bias Mitigation Measures", "Usage Constraints", "Summary of Performance Assessment",
        "Disaggregated Performance", "Testing Contexts",
        "Evaluations for Edge Cases or Adversarial Inputs", "Potential Risks and Harms",
        "Actions taken", "Misuse Scenarios", "Human Oversight", "Update Frequency"
    ]
    
    total = len(all_sections)
    base = total // 5
    remainder = total % 5

    groups = []
    start = 0
    for i in range(5):
        end = start + base + (1 if i < remainder else 0)
        groups.append(all_sections[start:end])
        start = end

    return groups  # returns [group1, group2, ..., group5]


