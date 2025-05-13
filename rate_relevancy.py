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
from utils.section_summary import generate_section_summary
from utils.top_level_summary import generate_top_level_summary
from utils.interactive_heatmap import generate_interactive_heatmap
from utils.policy_summary import generate_policy_summary
from utils.parse_model_card_content import parse_model_card_content

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

async def rate_relevancy():
        policy_folder = 'policies'
        policy_files = sorted(os.listdir(policy_folder))
        prompt_template_path = "relevancy_prompt.txt"
        
        async with aiofiles.open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = await f.read()
        
        for policy_file in policy_files:
            try:
                policy_path = os.path.join(policy_folder, policy_file)
                async with aiofiles.open(policy_path, "r") as pf:
                    legal_doc_content = await pf.read()
                
                # Get chunking strategy for this policy
                chunking_prompt = get_chunking_prompt().replace("{{POLICY_DOC}}", legal_doc_content)
                chunk_response = llm.invoke(chunking_prompt).content
                chunks = parse_chunk_response(chunk_response)
                print(f"Policy {policy_file} will be evaluated in {len(chunks)} chunks")
                print("Chunks:", chunks)

                # Initialize section data for this policy
                policy_section_scores = {section: {} for section in sections}
                policy_section_descriptions = {section: {} for section in sections}

                for section in sections:
                        for chunk_start, chunk_end in chunks:
                            chunk_prompt = (
                                prompt_template
                                .replace("{{MODEL_CARD}}", model_card_content)
                                .replace("{{LEGAL_DOC}}", legal_doc_content)
                                .replace("{{SECTION}}", section)
                                .replace("{{START_ART}}", str(chunk_start))
                                .replace("{{END_ART}}", str(chunk_end))
                            )
                            print(f"Evaluating {policy_file} section '{section}' for articles {chunk_start}-{chunk_end}...")
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

                        # Parse the markdown table to extract JSON content
                        try:
                            # Split the response into lines and find all data rows
                            lines = response.content.strip().splitlines()
                            # Get all rows except the separator row (the one with |---|---|)
                            table_rows = [line for line in lines if line.startswith('|') and not line.startswith('|-')]
                            if len(table_rows) < 2:  # Need at least header and one data row
                                raise Exception("Invalid table format - missing header or data rows")
                            
                            # Skip header row (first row) and process each data row
                            data_rows = table_rows[1:]  # Skip header row
                            
                            for data_row in data_rows:
                                # Split row into cells and remove empty cells at start/end
                                cells = [cell.strip() for cell in data_row.split('|')[1:-1]]
                                if len(cells) != 2:  # Should have exactly 2 columns
                                    print(f"Warning: Row does not have 2 columns: {data_row}")
                                    continue
                                    
                                try:
                                    # First cell should be the article number
                                    # Clean and standardize the article number format
                                    article_num = cells[0].strip()
                                    # Remove any 'Art.' prefix if it exists
                                    article_num = article_num.replace('Art.', '').strip()
                                    # Keep the original number format (don't convert to int)
                                    
                                    # Second cell should be the JSON data
                                    json_data = json.loads(cells[1])
                                    
                                    policy_section_scores[section][article_num] = json_data['score']
                                    policy_section_descriptions[section][article_num] = json_data.get('description', '')
                                    print(f"Parsed article {article_num}: Score={json_data['score']}")
                                except (ValueError, json.JSONDecodeError) as e:
                                    print(f"Error parsing row {data_row}: {e}")
                                    continue
                        except Exception as e:
                            print(f"Error processing response: {e}")
                            print(f"Full response:\n{response}")

                # Store the data for this policy
                policy_name = policy_file.split('.')[0]
                
                # Get all article numbers and convert to float for proper sorting
                all_articles = set()
                for section in sections:
                    all_articles.update(policy_section_scores[section].keys())
                    
                print(f"\nDebug - Raw articles for {policy_name}:", all_articles)
                
                # Convert to float for sorting, handling both integer and decimal article numbers
                def article_to_sortable(art):
                    # Remove any 'Art.' prefix if it exists
                    clean_art = art.replace('Art.', '').strip()
                    try:
                        # Try converting to float to handle both integer and decimal article numbers
                        return float(clean_art)
                    except ValueError:
                        # If conversion fails, return the original string
                        print(f"Warning: Could not convert article number '{clean_art}' to float")
                        return clean_art
                        
                # Sort articles using the custom sorting function
                sorted_articles = sorted(all_articles, key=article_to_sortable)                
                all_policy_data[policy_name] = {
                    'scores': policy_section_scores,
                    'descriptions': policy_section_descriptions,
                    'articles': sorted_articles
                }

                # After processing each policy, organize data by section
                for section in sections:
                    section_data[section][policy_name] = {
                        'scores': policy_section_scores[section],
                        'descriptions': policy_section_descriptions[section]
                    }
                

            except Exception as e:
                print(f"Error processing policy file {policy_file}: {str(e)}")
                continue
