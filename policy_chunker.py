import json

def get_chunking_prompt():
    return """Analyze the following markdown table policy document and split it into evaluation chunks.

Return ONLY a JSON array of chunk boundaries in the format:
[
  { "start": "Article X", "end": "Article Y" },
  ...
]

Rules:
1. An "Article" is defined as **all rows that share the same first column value** (e.g., "Article 3"). 
   - **Do not** treat each row as a separate article. For example, if "Article 3" appears in 68 rows, it still counts as **one single Article**.
2. Each chunk must contain between 10 and 20 such Articles (not rows).
3. Keep related Articles together if possible.

Return only the JSON array. No extra text.

Policy Document:
{{POLICY_DOC}}
"""

def parse_chunk_response(response):
    """Parse Claude's chunking response into a list of (start, end) tuples."""
    try:
        chunks = json.loads(response)
        return [(chunk["start"], chunk["end"]) for chunk in chunks]
    except Exception as e:
        print(f"Error parsing chunk response: {e}")
        # Fallback to evaluating everything at once
        return [(1, 999)]  # Large end number to include all articles 