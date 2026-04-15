# Policy Annotator Website

A web app and Python pipeline for evaluating model card content against AI policy documents (GDPR, EU AI Act, CCPA, AIDA, Colorado), then generating:

- policy-level compliance scores and explanations
- section summaries and top-level summary
- interactive heatmap outputs

## What This Repo Contains

- `app.py` - Quart web server and upload endpoint
- `ai_pipeline.py` - core async evaluation pipeline
- `policies/` - policy text files used for evaluation
- `relevancy_maps/` - section-to-article mappings per policy
- `templates/` - frontend HTML templates
- `utils/` - summary + visualization helpers
- `uploads/`, `reports/`, `static/`, `cost_reports/` - generated runtime artifacts

## Prerequisites

- Python 3.10+ (3.11 or 3.12 recommended)
- An Anthropic API key

## 1) Clone and Create Virtual Environment

```bash
git clone <your-repo-url>
cd policy_annotator_website
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install Dependencies

This project imports more packages than currently listed in `requirements.txt`, so install both:

```bash
pip install -r requirements.txt
pip install quart aiofiles python-dotenv langchain-anthropic seaborn matplotlib numpy
```

If you want to make this reproducible for everyone, update `requirements.txt` with the same packages.

## 3) Configure Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Notes:

- `.env` is gitignored and should not be committed.
- If the key is missing/invalid, evaluations will fail when LLM calls are made.

## 4) Run the Web App

```bash
python3 app.py
```

Then open:

[http://localhost:8000](http://localhost:8000)

## How to Use the UI

1. Open the homepage.
2. Upload a model card CSV file.
3. Optionally select which policies to run.
4. Submit and wait for evaluation to complete.
5. Review returned heatmaps and summaries.

## Expected CSV Input Format

Your uploaded CSV should include columns:

- `Section`
- `Your Response`

Rows with missing values are skipped during model card markdown conversion.

## Output Files

At runtime, the app creates/updates:

- `uploads/` - uploaded CSV files (UUID-prefixed)
- `reports/` - generated report text files
- `static/` - generated heatmap HTML files
- `cost_reports/` - token/cost markdown reports

## Running Pipeline Directly (Without Web UI)

You can call `run_ai_pipeline(...)` from Python:

```python
import asyncio
from ai_pipeline import run_ai_pipeline

async def main():
    heatmap_files, summaries, top_level_summary, section_summaries = await run_ai_pipeline(
        model_card_path="uploads/example.csv",
        policy_folder="policies",
        output_path="reports/example_report.txt",
        selected_policies=["GDPR", "EU"]  # optional
    )
    print(heatmap_files)

asyncio.run(main())
```

## Other Utility Scripts

Useful scripts in this repo include:

- `parse_relevancy_rating.py`
- `process_relevancy_ratings.py`
- `filter_excel_by_max_score.py`
- `generate_irrelevancy_map.py`

Run any script directly, for example:

```bash
python3 process_relevancy_ratings.py
```

## Common Issues

- **`ModuleNotFoundError` for Quart / langchain / aiofiles**
  - Install missing packages in the active virtual environment.
- **`ANTHROPIC_API_KEY` missing**
  - Set it in `.env` and restart the app.
- **Upload rejected**
  - Only `.csv` uploads are accepted by `app.py`.
- **No policies selected / bad names**
  - `selected_policies` must match policy filenames without `.txt` (for example `GDPR`, `EU`, `CCPA`).

## Development Notes

- Main app entrypoint: `app.py`
- Server port: `8000`
- Policy files are loaded from `policies/`
- Relevancy mappings are loaded from `relevancy_maps/*_relevancy_map.py`

## Security

- Never commit real API keys.
- Rotate keys if they are ever exposed in commit history.
