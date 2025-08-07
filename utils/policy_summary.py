import os
import time
import json
import asyncio
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import aiofiles
import pandas as pd
import seaborn as sns

async def generate_policy_summary(policy_name, policy_data, model_card_content, llm, sections):
    """Generate a summary of policy compliance evaluation results"""
    try:
        print(f"Starting policy summary generation for {policy_name}")
        
        # Read the prompt template
        async with aiofiles.open("prompt_summarize.txt", "r") as f:
            prompt_template = await f.read()

        # Format the evaluation results for the prompt
        evaluation_results = []
        total_articles = 0
        non_compliant_articles = 0
        
        for section in sections:
            if section not in policy_data['scores']:
                print(f"Warning: Section '{section}' not found in policy data for {policy_name}")
                continue
                
            for article, score in policy_data['scores'][section].items():
                total_articles += 1
                if score < 5:  # Only include non-compliant items
                    non_compliant_articles += 1
                    description = policy_data['descriptions'][section].get(article, "No description available")
                    evaluation_results.append({
                        "section": section,
                        "article": article,
                        "score": score,
                        "description": description
                    })

        print(f"Policy {policy_name}: {total_articles} total articles, {non_compliant_articles} non-compliant articles")

        # If no non-compliant articles found, return a success message
        if not evaluation_results:
            return f"""**Current Compliance Status:**
The model card demonstrates full compliance with {policy_name} requirements. All evaluated sections meet the necessary standards with no identified compliance gaps.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| None | All requirements met | Continue monitoring compliance | N/A |"""

        # Format the evaluation results as a string
        evaluation_str = json.dumps(evaluation_results, indent=2)
        print(f"Evaluation results JSON length: {len(evaluation_str)} characters")

        # Prepare the prompt
        prompt = prompt_template.replace("{{POLICY_DOC_NAME}}", policy_name)
        prompt = prompt.replace("{{EVALUATION_RESULT}}", evaluation_str)
        
        print(f"Prompt length: {len(prompt)} characters")

        # Get summary from Claude with timeout protection
        try:
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, prompt),
                timeout=300  # 5 minute timeout
            )
            print(f"Successfully generated summary for {policy_name}")
            return response.content.strip()
        except asyncio.TimeoutError:
            print(f"Timeout error generating summary for {policy_name}")
            return f"""**Current Compliance Status:**
Timeout occurred while generating detailed summary for {policy_name}. Please check the evaluation results manually.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| Summary Generation Timeout | Unable to generate detailed summary | Review evaluation results manually | Medium |"""
        except Exception as e:
            print(f"Error calling LLM for {policy_name}: {str(e)}")
            return f"""**Current Compliance Status:**
Error occurred while generating summary for {policy_name}. Please check the evaluation results manually.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| Summary Generation Error | {str(e)} | Review evaluation results manually | Medium |"""
            
    except Exception as e:
        print(f"Error in generate_policy_summary for {policy_name}: {str(e)}")
        return f"""**Current Compliance Status:**
Error occurred while processing {policy_name} summary. Please check the evaluation results manually.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| Processing Error | {str(e)} | Review evaluation results manually | Medium |"""