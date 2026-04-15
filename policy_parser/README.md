# Policy Parser

This folder contains the policy parsing and evaluation workflow used to convert legal policy text into a structured clause table using Claude.

## Most Important Feature

**Convert policy text into a structured clause table** with columns:

- `Article Number`
- `Clause Number`
- `Content`

The main output files are:

- `policy_outputs/EU_table.txt`
- `policy_outputs/EU_table.md`
- `policy_outputs/EU_processing_report.md`

---

## Quick Start (Structured Articles Already Exist)

If `policy_outputs/Articles_structured/Article_*.txt` already exists, run:

```bash
cd policy_parser
python3 extract_policies.py
```

This reads all article files, sends them to Claude concurrently, and writes a single consolidated clause table.

---

## Full Pipeline (From Raw Policy Text)

If you are starting from raw policy text/HTML:

1. Prepare article files in:
   - `policy_outputs/Articles_structured/Article_1.txt`, `Article_2.txt`, ...
2. Ensure annexes text exists:
   - `policy_outputs/EU_AI_Act_Annexes.txt`
3. Run:

```bash
cd policy_parser
python3 extract_policies.py
```

If you want a simpler sequential extractor from one raw text file, you can use:

```bash
python3 process_eu_ai_act.py
```

`process_eu_ai_act.py` expects:

- `policy_outputs/EU AI Act.txt`
- `policy_outputs/EU_AI_Act_Annexes.txt`

---

## Environment Setup

### 1) Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install Dependencies

Install at minimum:

```bash
pip install langchain-anthropic python-dotenv pandas beautifulsoup4 tabulate
```

### 3) API Key

Create a `.env` in the project root or `policy_parser/`:

```env
ANTHROPIC_API_KEY=your_key_here
```

---

## Input and Output Details

### Primary Inputs for `extract_policies.py`

- `policy_outputs/Articles_structured/*.txt`  
  Each file should contain one article's text.
- `policy_outputs/EU_AI_Act_Annexes.txt`  
  Annexes content used as background context.

### Primary Outputs

- `policy_outputs/EU_table.txt` - combined table (text)
- `policy_outputs/EU_table.md` - combined table (markdown)
- `policy_outputs/EU_processing_report.md` - runtime + token/cost report

---

## How `extract_policies.py` Works

1. Loads annexes and all article files.
2. Sends each article to Claude with instructions to normalize into a table row format.
3. Processes articles concurrently (`ThreadPoolExecutor`).
4. Merges all responses into one table while preserving article order.
5. Writes final files and a processing report.

---

## Other Useful Scripts in This Folder

- `evaluate_model_card_claude.py`  
  Scores model cards against structured policy table output.
- `generate_policy_tables.py`  
  Aggregates evaluation outputs by policy.
- `generate_model_tables.py`  
  Aggregates evaluation outputs by model.
- `process_contribution_scores.py`  
  Builds one combined contribution matrix.
- `generate_model_card_claude.py`  
  Generates example model cards from templates.
- `clean_html.py`  
  Strips HTML to readable text.
- `recitals_annexes_extraction.py`  
  Extracts recitals/annexes from policy text using Claude.

---

## Common Issues

- **`KeyError: ANTHROPIC_API_KEY` / auth failures**  
  Ensure `.env` is loaded and key is valid.

- **No article files found**  
  Confirm files are under `policy_outputs/Articles_structured/` and named like `Article_12.txt`.

- **Rate limits / long runtime**  
  `extract_policies.py` is highly concurrent; reduce `max_workers` in script if needed.

- **Unexpected parse quality**  
  Refine the prompt inside `extract_policies.py` and rerun.

---

## Recommended Script Today

For clause-table conversion, prefer:

```bash
python3 extract_policies.py
```

It is the more complete and better-instrumented implementation in this folder.
