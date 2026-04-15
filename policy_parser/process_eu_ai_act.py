import os
import time
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import re

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0.3
)

# Track costs and time
start_time = time.time()
total_input_tokens = 0
total_output_tokens = 0
api_calls = 0

# Pricing for Claude 3.5 Sonnet (as of 2024)
INPUT_COST_PER_1K = 0.003  # $3 per 1M tokens
OUTPUT_COST_PER_1K = 0.015  # $15 per 1M tokens

def calculate_cost(input_tokens, output_tokens):
    """Calculate API cost based on token usage."""
    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
    return input_cost + output_cost

def load_reference_documents(annexes_file):
    """Loads Annexes into memory for use as a dictionary."""
    if os.path.exists(annexes_file):
        with open(annexes_file, "r", encoding="utf-8") as f:
            annexes_text = f.read().strip()
        return annexes_text
    return ""

# Removed unused functions - using regex extraction instead

def extract_policy_clauses(article_text, annexes_text):
    """
    Extracts policy clauses from a single article document while referring to Annexes.
    """
    global total_input_tokens, total_output_tokens, api_calls
    
    prompt = f"""
    #You will be provided with
        1, An article of a Policy Document (Current Articles for Processing)
        2, Background information: Annexes (Technical requirements, classifications, and compliance details)

    #Instruction
        1. Structure the policy document into a markdown table with 3 columns: Article No., Clause No. within article, and Content.
        2. An Article starts with the title "Article Number" such as "Article 1", "Article 2", etc.
        3. A policy clause is a bullet point under an article. It consists of a number and its content. For example, "1.Providers of high-risk AI systems shall, upon a reasoned request by a competent authority, provide that authority all the information and documentation necessary to demonstrate the conformity of the high-risk AI system with the requirements set out in Section 2, in a language which can be easily understood by the authority in one of the official languages of the institutions of the Union as indicated by the Member State concerned." is a policy clause under Article 21. 
        4, IMPORTANT: If a clause refers to certain parts of Annex for context or further information, add that information in parentheses () after that reference. For example, input with "in accordance with Article 11 and Annex IV" outputs "in accordance with Article 11 and Annex IV(Fill with content from Annex IV)", input with "product pursuant to the Union harmonisation legislation listed in Annex I" outputs product pursuant to the Union harmonisation legislation listed in Annex I(Fill with content from Annex I that talks about Union harmonisation legislation).You should not modify the policy clause's content in any other way.
        5, If a clause has bullet points underneath it, for example: "2.The AI Office and the Board shall aim to ensure that the codes of practice cover at least the obligations provided for in Articles 53 and 55, including the following issues: (a)the means to ensure that the information referred to in Article 53(1), points (a) and (b), is kept up to date in light of market and technological developments;(b)the adequate level of detail for the summary about the content used for training;(c)the identification of the type and nature of the systemic risks at Union level, including their sources, where appropriate;" point a), b), c) should be appended right after "following issues: " and this would fit in one row(counts as one clause), instead of multiple. 
        6. Process the ENTIRE input article without asking for permission to continue or stopping mid-article.

    #Annexes (Background Info)
    {annexes_text[:10000]}  # Limit annexes text to avoid token limits

    #A section of Policy Document (Current Article for Processing)
    {article_text}

    # IMPORTANT: continue with all articles even when the output is lengthy. Your response should be only consisted of the markdown table, no extra question/note/description should be output.
    """

    try:
        response = llm.invoke(prompt, max_tokens=8192)
        api_calls += 1
        
        # Track tokens
        estimated_input_tokens = len(prompt.split()) * 1.3
        estimated_output_tokens = len(response.content.split()) * 1.3
        total_input_tokens += estimated_input_tokens
        total_output_tokens += estimated_output_tokens
        
        return response.content
    except Exception as e:
        print(f"❌ Error extracting policy clauses: {e}")
        return ""

def append_extracted_content(extracted_text, output_file, is_first_article):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Split text into rows
    lines = extracted_text.strip().split("\n")

    # Keep the header for the first article, remove it for subsequent articles
    if not is_first_article and len(lines) > 1:
        lines = lines[2:]  # Remove header row for subsequent articles

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Appended content to: {output_file}")

def save_evaluation_report(output_file, start_time, total_input_tokens, total_output_tokens, api_calls, articles_processed):
    """Save evaluation time and cost report."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_cost = calculate_cost(total_input_tokens, total_output_tokens)
    
    report = f"""# EU AI Act Processing Evaluation Report

## Processing Details
- **Start Time**: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
- **End Time**: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}
- **Total Processing Time**: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)
- **Articles Processed**: {articles_processed}

## API Usage
- **Total API Calls**: {api_calls}
- **Model Used**: claude-3-5-sonnet-20241022
- **Temperature**: 0.3

## Token Usage (Estimated)
- **Total Input Tokens**: {total_input_tokens:,.0f}
- **Total Output Tokens**: {total_output_tokens:,.0f}
- **Total Tokens**: {total_input_tokens + total_output_tokens:,.0f}

## Cost Estimation
- **Input Cost**: ${(total_input_tokens / 1000) * INPUT_COST_PER_1K:.4f}
- **Output Cost**: ${(total_output_tokens / 1000) * OUTPUT_COST_PER_1K:.4f}
- **Total Estimated Cost**: ${total_cost:.4f}

## Pricing Reference
- Input: ${INPUT_COST_PER_1K * 1000:.3f} per 1M tokens
- Output: ${OUTPUT_COST_PER_1K * 1000:.3f} per 1M tokens

## Notes
- Token counts are estimated based on word count (multiplied by 1.3)
- Actual costs may vary based on actual token usage from API responses
- Processing includes article extraction and clause structuring
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📊 Evaluation report saved to: {output_file}")
    print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
    print(f"💰 Estimated cost: ${total_cost:.4f}")

if __name__ == "__main__":
    # File paths
    INPUT_FILE = "policy_outputs/EU AI Act.txt"
    ANNEXES_FILE = "policy_outputs/EU_AI_Act_Annexes.txt"
    OUTPUT_TXT_FILE = "policy_outputs/EU_table.txt"
    OUTPUT_MD_FILE = "policy_outputs/EU_table.md"
    REPORT_FILE = "policy_outputs/EU_processing_report.md"
    
    print("🚀 Starting EU AI Act processing...")
    print(f"📖 Reading input file: {INPUT_FILE}")
    
    # Load raw text
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"📄 File size: {len(raw_text):,} characters")
    
    # Load annexes
    annexes_text = load_reference_documents(ANNEXES_FILE)
    print(f"📋 Loaded annexes: {len(annexes_text):,} characters")
    
    # Clear existing output files
    open(OUTPUT_TXT_FILE, "w").close()
    open(OUTPUT_MD_FILE, "w").close()
    
    # For now, we'll use a simpler approach: process the text directly
    # Since the file is very large, we'll extract articles using regex first
    # Then process each article
    
    # Extract articles using regex pattern - look for "Article X" on its own line
    article_pattern = r'Article\s+(\d+)\s*\n(.*?)(?=\nArticle\s+\d+\s*\n|$)'
    articles = re.findall(article_pattern, raw_text, re.DOTALL)
    
    print(f"📑 Found {len(articles)} articles")
    
    if not articles:
        print("⚠️  No articles found with regex pattern. Trying alternative pattern...")
        # Try alternative pattern
        article_pattern = r'Article\s+(\d+)(.*?)(?=Article\s+\d+|$)'
        articles = re.findall(article_pattern, raw_text, re.DOTALL)
        print(f"📑 Found {len(articles)} articles with alternative pattern")
    
    if articles:
        articles_list = [{'number': f'Article {num}', 'content': content.strip()} for num, content in articles]
    else:
        print("⚠️  No articles found. Please check the file format.")
        articles_list = []
    
    # Process each article
    is_first_article = True
    articles_processed = 0
    
    # Remove limit for full processing - process all articles
    for i, article in enumerate(articles_list, 1):
        print(f"\n📄 Processing {article['number']} ({i}/{len(articles_list)})...")
        
        article_text = f"{article['number']}\n\n{article['content']}"
        extracted_text = extract_policy_clauses(article_text, annexes_text)
        
        if extracted_text:
            append_extracted_content(extracted_text, OUTPUT_TXT_FILE, is_first_article)
            append_extracted_content(extracted_text, OUTPUT_MD_FILE, is_first_article)
            is_first_article = False
            articles_processed += 1
        
        # Add delay to avoid rate limiting
        if i < len(articles_list):
            time.sleep(10)
    
    # Save evaluation report
    save_evaluation_report(REPORT_FILE, start_time, total_input_tokens, total_output_tokens, api_calls, articles_processed)
    
    print(f"\n✅ Processing complete! Processed {articles_processed} articles.")
    print(f"📁 Output files:")
    print(f"   - {OUTPUT_TXT_FILE}")
    print(f"   - {OUTPUT_MD_FILE}")
    print(f"   - {REPORT_FILE}")

