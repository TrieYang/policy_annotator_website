import os
import time
import json
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import aiofiles
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from policy_chunker import get_chunking_prompt, parse_chunk_response
import asyncio

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
model="claude-sonnet-4-20250514",
#model="claude-3-haiku-20240307",
anthropic_api_key=ANTHROPIC_API_KEY,
temperature=0.3)

# Section names for iteration
sections = [
    "System Name",
    "Versioning Information",
    "Primary Developer/Org",
    "Contact Info",
    "System Overview",
    "Primary intended uses",
    "Primary intended users",
    "Out-of-scope use cases", 
    "Terms and conditions",  
    "Current legal compliance status", 
    "Dataset Description",
    "Collection Method",
    "Bias Mitigation Measures",
    "Usage Constraints",
    "Summary of Performance Assessment",
    "Disaggregated Performance", 
    "Testing Contexts",
    "Evaluations for Edge Cases or Adversarial Inputs",
    "Potential Risks and Harms",
    "Actions taken",
    "Misuse Scenarios",
    "Human Oversight", 
    "Update Frequency" 
]

sections_pairs = [
    ["System Name", "Contact Info"],
    ["System Overview","Terms and conditions"],
    ["Current legal compliance status", "Bias Mitigation Measures"],
    ["Usage Constraints","Testing Contexts"],
    ["Evaluations for Edge Cases or Adversarial Inputs", "Actions taken"],
    ["Misuse Scenarios", "Update Frequency"]
]

async def rate_relevancy():
        policy_folder = 'unprocessed_policy_tables'
        policy_files = sorted(os.listdir(policy_folder))
        prompt_template_path = "relevancy_prompt.txt"
        model_card_path = "model_cards/mc3.txt"  
        
        async with aiofiles.open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = await f.read()
            
        # Read the model card content
        async with aiofiles.open(model_card_path, "r", encoding="utf-8") as f:
            model_card_content = await f.read()
        
        # Open the output file for appending responses
        async with aiofiles.open("relevancy_rating.txt", "a", encoding="utf-8") as output_file:
            for policy_file in policy_files:
                try:
                    policy_path = os.path.join(policy_folder, policy_file)
                    async with aiofiles.open(policy_path, "r") as pf:
                        legal_doc_content = await pf.read()
                    
                    # Get chunking strategy for this policy
                    chunking_prompt = get_chunking_prompt(irre=False).replace("{POLICY_DOC}", legal_doc_content)
                    chunk_response = llm.invoke(chunking_prompt).content
                    print(chunk_response)
                    chunks = parse_chunk_response(chunk_response, irre=False)
                    print(f"Policy {policy_file} will be evaluated in {len(chunks)} chunks")
                    print("Chunks:", chunks)

                    for section_pair in sections_pairs:
                            for chunk_start, chunk_end in chunks:
                                chunk_prompt = (
                                    prompt_template
                                    .replace("{{MODEL_CARD}}", model_card_content)
                                    .replace("{{START_SECTION}}", section_pair[0])
                                    .replace("{{END_SECTION}}", section_pair[1])
                                    .replace("{{START_ART}}", str(chunk_start))
                                    .replace("{{END_ART}}", str(chunk_end))
                                )
                                print(f"Evaluating {policy_file} section '{section_pair[0]}' to '{section_pair[1]}' for articles {chunk_start}-{chunk_end}...")
                                messages = [
                                    {
                                        "role": "system",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": f"{legal_doc_content}",
                                                "cache_control": {"type": "ephemeral"},
                                            },
                                        ],
                                    },
                                    {
                                        "role": "user",
                                        "content": f"{chunk_prompt}",
                                    },
                                ]
                                response = llm.invoke(messages)
                                print(response)
                                
                                # Append the response to the output file
                                await output_file.write(f"\n{'='*80}\n")
                                await output_file.write(f"Policy: {policy_file}\n")
                                await output_file.write(f"Articles: {chunk_start}-{chunk_end}\n")
                                await output_file.write(f"Response:\n{response.content}\n")

                except Exception as e:
                    print(f"Error processing policy file {policy_file}: {str(e)}")
                    continue
# Example usage
if __name__ == "__main__":
    asyncio.run(rate_relevancy())
