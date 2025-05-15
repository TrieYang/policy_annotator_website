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
2. Each chunk must contain between 5 and 10 such Articles (not rows).
3. Keep related Articles together if possible.
4. You will also be given a list of irrelevant articles that doesn't need to be included in the chunk boundaries. for example, if 
   {"AIDA.Art.1",
    "AIDA.Art.3",
    "AIDA.Art.6",
    "AIDA.Art.10",
    "AIDA.Art.13",
    "AIDA.Art.18",
    "AIDA.Art.19",
    "AIDA.Art.20",
    "AIDA.Art.21",
    "AIDA.Art.22",
    "AIDA.Art.23",
    "AIDA.Art.24",
    "AIDA.Art.25" }
    is given, your chunking would avoid any of these articles.

Return only the JSON array. 
**Do not include any extra text than a single JSON array**.

Policy Document:
{{POLICY_DOC}}
Irrelevant Article List:
{{IRRE_LIST}}
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