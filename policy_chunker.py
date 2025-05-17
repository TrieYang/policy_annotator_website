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

Rules:
1. An "Article" is defined as **all rows that share the same first column value** (e.g., "Article 3"). 
   - **Do not** treat each row as a separate article. For example, if "Article 3" appears in 68 rows, it still counts as **one single Article**.
2. Each chunk must contain between 5 and 10 such Articles (not rows).
3. Keep related Articles together if possible.

4. **CRITICAL - Handling Irrelevant Articles:**
   - You will receive a list of irrelevant articles for each section
   - You MUST NOT include any irrelevant articles in your chunk boundaries
   - If an article is in the irrelevant list, you must skip it entirely
   - Example 1: If a policy has Articles 1-5 and Article 3 is irrelevant, valid chunks would be:
     * {{"s": "Article 1", "e": "Article 2"}}
     * {{"s": "Article 4", "e": "Article 5"}}
   - Example 2: If a policy has Articles 1-10 and Articles 3,7 are irrelevant, valid chunks would be:
     * {{"s": "Article 1", "e": "Article 2"}}
     * {{"s": "Article 4", "e": "Article 6"}}
     * {{"s": "Article 8", "e": "Article 10"}}

5. Make sure to include ALL relevant articles that are not in the irrelevant list.
6. If a section has no relevant articles (all are irrelevant), return an empty array for that section.
7. Only process the following sections: {sections_str}
8. 
Policy Document: {{POLICY_DOC}}
Section-Specific Irrelevant Article Lists: {{IRRE_LIST}}
Return only the JSON object. 
**Do not include any extra text than a single JSON object**.
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