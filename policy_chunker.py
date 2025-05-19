import json

def get_chunking_prompt(sections=None):
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

def parse_chunk_response(response):
    """Parse Claude's chunking response into a dictionary of section-specific chunk boundaries."""
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

def get_section_groups():
    """Split sections into two groups for processing."""
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
    
    # Split sections into two roughly equal groups
    mid_point = len(all_sections) // 2
    group1 = all_sections[:mid_point]
    group2 = all_sections[mid_point:]
    
    return group1, group2 