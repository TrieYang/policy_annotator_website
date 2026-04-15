import os
import time
import pandas as pd
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Track costs and time (thread-safe)
start_time = time.time()
total_input_tokens = 0
total_output_tokens = 0
api_calls = 0
token_lock = threading.Lock()  # Lock for thread-safe token tracking

# Pricing for Claude 3.5 Sonnet (as of 2024)
INPUT_COST_PER_1K = 0.003  # $3 per 1M tokens
OUTPUT_COST_PER_1K = 0.015  # $15 per 1M tokens

def calculate_cost(input_tokens, output_tokens):
    """Calculate API cost based on token usage."""
    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
    return input_cost + output_cost

def load_reference_documents(annexes_file, main_folder):
    """Loads Annexes, and all Main articles into memory for use as a dictionary."""

    with open(annexes_file, "r", encoding="utf-8") as f:
        annexes_text = f.read().strip()

    main_articles_text = ""
    for filename in sorted(os.listdir(main_folder)): 
        if filename.endswith(".txt"):
            with open(os.path.join(main_folder, filename), "r", encoding="utf-8") as f:
                main_articles_text += f"### {filename} ###\n{f.read().strip()}\n\n"

    return annexes_text, main_articles_text

def extract_policy_clauses(article_text, annexes_text, main_articles_text, article_name="", article_index=0):
    """
    Extracts policy clauses from a single article document while referring to Annexes,
    and other Articles for definitions/context.
    Creates its own LLM instance for thread safety.
    """
    global total_input_tokens, total_output_tokens, api_calls
    
    # Create a new LLM instance for this thread (thread-safe)
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", anthropic_api_key=ANTHROPIC_API_KEY)
    
    prompt = f"""
    #You will be provided with
        1, An article of a Policy Document (Current Articles for Processing)
        2, Background information: Annexes (Technical requirements, classifications, and compliance details)

    #Instruction
        1. Structure the policy document into a markdown table with 3 columns: Article No., Clause No. within article, and Content.
        2. An Article starts with the title "Article Number" such as "Article 1", "Article 2", etc.
        3. A policy clause is a bullet point under an article. It consists of a number and its content. Dor example, "1.Providers of high-risk AI systems shall, upon a reasoned request by a competent authority, provide that authority all the information and documentation necessary to demonstrate the conformity of the high-risk AI system with the requirements set out in Section 2, in a language which can be easily understood by the authority in one of the official languages of the institutions of the Union as indicated by the Member State concerned." is a policy clause under Article 21. 
        4, IMPORTANT: If a clause refers to certain parts of Annex for context or further information, add that information in parentheses () after that reference. For example, intput with "in accordance with Article 11 and Annex IV" outputs "in accordance with Article 11 and Annex IV(Fill with content from Annex IV)", input with "product pursuant to the Union harmonisation legislation listed in Annex I" outputs product pursuant to the Union harmonisation legislation listed in Annex I(Fill with content from Annex I that talks about Union harmonisation legislation).You should not modify the policy clause's content in any other way.
        6, If a clause has bullet points underneath it, for example: "2.The AI Office and the Board shall aim to ensure that the codes of practice cover at least the obligations provided for in Articles 53 and 55, including the following issues: (a)the means to ensure that the information referred to in Article 53(1), points (a) and (b), is kept up to date in light of market and technological developments;(b)the adequate level of detail for the summary about the content used for training;(c)the identification of the type and nature of the systemic risks at Union level, including their sources, where appropriate;" point a), b), c) should be appended right after "following issues: " and this would fit in one row(counts as one clause), instead of multiple. 
        7. Process the ENTIRE input article without asking for permission to continue or stopping mid-article.

    #Examples
    ##Input 1(this is a basic example): 
        Article 21

        Cooperation with competent authorities

        1.Providers of high-risk AI systems shall, upon a reasoned request by a competent authority, provide that authority all the information and documentation necessary to demonstrate the conformity of the high-risk AI system with the requirements set out in Section 2, in a language which can be easily understood by the authority in one of the official languages of the institutions of the Union as indicated by the Member State concerned.

        2.Upon a reasoned request by a competent authority, providers shall also give the requesting competent authority, as applicable, access to the automatically generated logs of the high-risk AI system referred to in Article 12(1), to the extent such logs are under their control.
    ##Output 1:
        | Article Number | Clause Number | Content |
        | -------------- | ------------- | ------- |
        | Article 21 | 1 | Providers of high-risk AI systems shall, upon a reasoned request by a competent authority, provide that authority all the information and documentation necessary to demonstrate the conformity of the high-risk AI system with the requirements set out in Section 2, in a language which can be easily understood by the authority in one of the official languages of the institutions of the Union as indicated by the Member State concerned. | 
        Article 21 | 2 | Upon a reasoned request by a competent authority, providers shall also give the requesting competent authority, as applicable, access to the automatically generated logs of the high-risk AI system referred to in Article 12(1), to the extent such logs are under their control. |
        
    ##Input 2(this is an example where it references an annex):
        Article 23

        Obligations of importers

        1.Before placing a high-risk AI system on the market, importers shall ensure that the system is in conformity with this Regulation by verifying that:

        (a)

        the relevant conformity assessment procedure referred to in Article 43 has been carried out by the provider of the high-risk AI system;

        (b)

        the provider has drawn up the technical documentation in accordance with Article 11 and Annex IV;

        (c)

        the system bears the required CE marking and is accompanied by the EU declaration of conformity referred to in Article 47 and instructions for use;

        (d)

        the provider has appointed an authorised representative in accordance with Article 22(1).
    ##Output 2:
        | Article Number | Clause Number | Content |
        | -------------- | ------------- | ------- |
        | Article 23| 1 | 1.Before placing a high-risk AI system on the market, importers shall ensure that the system is in conformity with this Regulation by verifying that:(a)the relevant conformity assessment procedure referred to in Article 43 has been carried out by the provider of the high-risk AI system;(b)the provider has drawn up the technical documentation in accordance with Article 11 and Annex IV(The technical documentation referred to in Article 11(1) shall contain at least the following information, as applicable to the relevant AI system: 1.A general description of the AI system including:(a)its intended purpose, the name of the provider and the version of the system reflecting its relation to previous versions;(b)how the AI system interacts with, or can be used to interact with, hardware or software, including with other AI systems, that are not part of the AI system itself, where applicable;(c)the versions of relevant software or firmware, and any requirements related to version updates;(d)the description of all the forms in which the AI system is placed on the market or put into service, such as software packages embedded into hardware, downloads, or APIs;(e)the description of the hardware on which the AI system is intended to run;(f)where the AI system is a component of products, photographs or illustrations showing external features, the marking and internal layout of those products;(g)a basic description of the user-interface provided to the deployer;(h)instructions for use for the deployer, and a basic description of the user-interface provided to the deployer, where applicable;2.A detailed description of the elements of the AI system and of the process for its development, including:(a)the methods and steps performed for the development of the AI system, including, where relevant, recourse to pre-trained systems or tools provided by third parties and how those were used, integrated or modified by the provider;(b)the design specifications of the system, namely the general logic of the AI system and of the algorithms; the key design choices including the rationale and assumptions made, including with regard to persons or groups of persons in respect of who, the system is intended to be used; the main classification choices; what the system is designed to optimise for, and the relevance of the different parameters; the description of the expected output and output quality of the system; the decisions about any possible trade-off made regarding the technical solutions adopted to comply with the requirements set out in Chapter III, Section 2;(c)the description of the system architecture explaining how software components build on or feed into each other and integrate into the overall processing; the computational resources used to develop, train, test and validate the AI system;(d)where relevant, the data requirements in terms of datasheets describing the training methodologies and techniques and the training data sets used, including a general description of these data sets, information about their provenance, scope and main characteristics; how the data was obtained and selected; labelling procedures (e.g. for supervised learning), data cleaning methodologies (e.g. outliers detection);(e)assessment of the human oversight measures needed in accordance with Article 14, including an assessment of the technical measures needed to facilitate the interpretation of the outputs of AI systems by the deployers, in accordance with Article 13(3), point (d);(f)where applicable, a detailed description of pre-determined changes to the AI system and its performance, together with all the relevant information related to the technical solutions adopted to ensure continuous compliance of the AI system with the relevant requirements set out in Chapter III, Section 2;(g)the validation and testing procedures used, including information about the validation and testing data used and their main characteristics; metrics used to measure accuracy, robustness and compliance with other relevant requirements set out in Chapter III, Section 2, as well as potentially discriminatory impacts; test logs and all test reports dated and signed by the responsible persons, including with regard to pre-determined changes as referred to under point (f);(h)cybersecurity measures put in place;3.Detailed information about the monitoring, functioning and control of the AI system, in particular with regard to: its capabilities and limitations in performance, including the degrees of accuracy for specific persons or groups of persons on which the system is intended to be used and the overall expected level of accuracy in relation to its intended purpose; the foreseeable unintended outcomes and sources of risks to health and safety, fundamental rights and discrimination in view of the intended purpose of the AI system; the human oversight measures needed in accordance with Article 14, including the technical measures put in place to facilitate the interpretation of the outputs of AI systems by the deployers; specifications on input data, as appropriate;4.A description of the appropriateness of the performance metrics for the specific AI system;5.A detailed description of the risk management system in accordance with Article 9;6.A description of relevant changes made by the provider to the system through its lifecycle;7.A list of the harmonised standards applied in full or in part the references of which have been published in the  Official Journal of the European Union ; where no such harmonised standards have been applied, a detailed description of the solutions adopted to meet the requirements set out in Chapter III, Section 2, including a list of other relevant standards and technical specifications applied;Official Journal of the European Union. 8.A copy of the EU declaration of conformity referred to in Article 47;9.A detailed description of the system in place to evaluate the AI system performance in the post-market phase in accordance with Article 72, including the post-market monitoring plan referred to in Article 72(3).);(c)the system bears the required CE marking and is accompanied by the EU declaration of conformity referred to in Article 47 and instructions for use;(d)the provider has appointed an authorised representative in accordance with Article 22(1).

    #Annexes (Background Info)
    {annexes_text}

    #A section of Policy Document (Current Article for Processing)**
    {article_text}

    # IMPORTANT: continue with all articles even when the output is lengthy. Your response should be only consisted of the markdown table, no extra question/note/description should be output.
    """

    try:
        call_start = time.time()
        response = llm.invoke(prompt, max_tokens=8192)
        call_duration = time.time() - call_start
        
        # Get actual token usage from API response
        input_tokens = 0
        output_tokens = 0
        
        # Try to get actual usage from response_metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('usage', {})
            if isinstance(usage, dict):
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
        
        # Fallback to estimation if not available
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = int(len(prompt.split()) * 1.3)
            output_tokens = int(len(response.content.split()) * 1.3)
            print(f"   [{article_index}] ⚠️  Using estimated tokens (actual usage not available)")
        else:
            print(f"   [{article_index}] ✓ Using actual token usage from API")
        
        # Thread-safe token tracking
        with token_lock:
            api_calls += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            current_total_input = total_input_tokens
            current_total_output = total_output_tokens
        
        elapsed = time.time() - start_time
        current_cost = calculate_cost(current_total_input, current_total_output)
        print(f"   [{article_index}] ⏱️  API call: {call_duration:.1f}s | Tokens: {input_tokens:,} in, {output_tokens:,} out | Total: {elapsed/60:.1f}min | Cost: ${current_cost:.4f}")
        
        return response.content
    except Exception as e:
        print(f"   [{article_index}] ❌ Error extracting policy clauses: {e}")
        return ""

def append_extracted_content(extracted_text, output_file, is_first_article):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Split text into rows
    lines = extracted_text.strip().split("\n")

    # Keep the header for the first article, remove it for subsequent articles
    if not is_first_article and len(lines) > 1:
        lines = lines[2:]  # Modified: Remove first row for subsequent articles

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   ✅ Saved to {os.path.basename(output_file)}")

def save_evaluation_report(output_file, start_time, total_input_tokens, total_output_tokens, api_calls, articles_processed, max_workers=None):
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
- **Model Used**: claude-sonnet-4-20250514
- **Temperature**: 0.3 (default)

## Token Usage
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
- Token counts are from actual API response when available, otherwise estimated
- Processing includes article extraction and clause structuring
- Articles processed with FULL concurrency (all articles processed simultaneously)
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📊 Evaluation report saved to: {output_file}")
    print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
    print(f"💰 Estimated cost: ${total_cost:.4f}")

# === 4. MAIN FUNCTION TO PROCESS ALL POLICY ARTICLES ===
if __name__ == "__main__":
    ANNEXES_FILE = "policy_outputs/EU_AI_Act_Annexes.txt"
    MAIN_FOLDER = "policy_outputs/Articles_structured"  
    OUTPUT_MD_FILE = "policy_outputs/EU_table.md"  
    OUTPUT_TXT_FILE = "policy_outputs/EU_table.txt"
    REPORT_FILE = "policy_outputs/EU_processing_report.md"

    print("=" * 70)
    print("🚀 Starting EU AI Act Processing")
    print("=" * 70)
    print(f"📂 Input folder: {MAIN_FOLDER}")
    print(f"📄 Annexes file: {ANNEXES_FILE}")
    print(f"💾 Output files: {OUTPUT_TXT_FILE}, {OUTPUT_MD_FILE}")
    print("=" * 70)
    
    annexes_text, main_articles_text = load_reference_documents(ANNEXES_FILE, MAIN_FOLDER)
    print(f"✅ Loaded annexes ({len(annexes_text):,} chars) and main articles reference")

    # Get list of article files
    article_files = [f for f in os.listdir(MAIN_FOLDER) if f.endswith(".txt") and f.startswith("Article_")]
    article_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if x.split('_')[1].split('.')[0].isdigit() else 999)
    
    total_articles = len(article_files)
    print(f"📑 Found {total_articles} articles to process")
    print("=" * 70)

    # Clear existing output files before writing new content
    open(OUTPUT_MD_FILE, "w").close() 
    open(OUTPUT_TXT_FILE, "w").close() 

    # Prepare article data for processing
    article_data = []
    for i, filename in enumerate(article_files, 1):
        article_path = os.path.join(MAIN_FOLDER, filename)
        article_num = filename.replace("Article_", "").replace(".txt", "")
        
        with open(article_path, "r", encoding="utf-8") as f:
            article_text = f.read().strip()
        
        article_data.append({
            'index': i,
            'filename': filename,
            'article_num': article_num,
            'article_text': article_text
        })
    
    print(f"🔄 Starting FULL concurrent processing (all {total_articles} articles at once)...")
    print("=" * 70)
    
    # Process articles concurrently
    results = {}  # Store results by index to maintain order
    articles_processed = 0
    max_workers = total_articles  # Process ALL articles concurrently
    
    def process_article(article_info):
        """Process a single article and return result with index."""
        idx = article_info['index']
        filename = article_info['filename']
        article_num = article_info['article_num']
        article_text = article_info['article_text']
        
        print(f"[{idx}/{total_articles}] 📄 Processing {filename}...")
        print(f"   [{idx}] ⏳ Starting API call...")
        
        extracted_text = extract_policy_clauses(
            article_text, annexes_text, main_articles_text, 
            article_num, idx
        )
        
        return {
            'index': idx,
            'article_num': article_num,
            'filename': filename,
            'extracted_text': extracted_text,
            'success': bool(extracted_text)
        }
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_article = {
            executor.submit(process_article, article): article 
            for article in article_data
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_article):
            result = future.result()
            results[result['index']] = result
            
            if result['success']:
                articles_processed += 1
                print(f"   [{result['index']}] ✅ Article {result['article_num']} completed ({articles_processed}/{total_articles} total processed)")
            else:
                print(f"   [{result['index']}] ⚠️  Article {result['article_num']} failed - no content extracted")
    
    # Write results in order
    print("\n" + "=" * 70)
    print("📝 Writing results to output files in order...")
    print("=" * 70)
    
    is_first_article = True
    for idx in sorted(results.keys()):
        result = results[idx]
        if result['success']:
            append_extracted_content(result['extracted_text'], OUTPUT_MD_FILE, is_first_article)
            append_extracted_content(result['extracted_text'], OUTPUT_TXT_FILE, is_first_article)
            is_first_article = False
    
    print("\n" + "=" * 70)
    print("✅ Processing Complete!")
    print("=" * 70)
    
    # Save evaluation report
    save_evaluation_report(REPORT_FILE, start_time, total_input_tokens, total_output_tokens, api_calls, articles_processed, max_workers)
    
    print(f"\n📁 Output files created:")
    print(f"   - {OUTPUT_TXT_FILE}")
    print(f"   - {OUTPUT_MD_FILE}")
    print(f"   - {REPORT_FILE}")
    print("=" * 70)

