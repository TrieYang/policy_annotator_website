import os
import time
import json
from dotenv import load_dotenv
import aiofiles

async def generate_section_summary(section_name, section_data, llm, TESTING_MODE):
    """Generate a summary of compliance evaluation results for a specific model card section"""
    
    # non-testing mode
    async with aiofiles.open("prompt_summarize_by_section.txt", "r") as f:
        prompt_template = await f.read()

    # Format the evaluation results for the prompt
    evaluation_results = []
    for policy_name, policy_data in section_data.items():
        for article, score in policy_data['scores'].items():
            if score < 5:  # Only include non-compliant items
                description = policy_data['descriptions'].get(article, "No description available")
                evaluation_results.append({
                    "policy": policy_name,
                    "article": article,
                    "score": score,
                    "description": description
                })

    # If no evaluation results, return a specific message for empty results
    if not evaluation_results:
        return json.dumps({
            "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\nNo compliance issues were identified for this section. All evaluated criteria meet the requirements."
        })

    # Format the evaluation results as a string
    evaluation_str = json.dumps(evaluation_results, indent=2)

    # Prepare the prompt
    prompt = prompt_template.replace("{{SECTION_NAME}}", section_name)
    prompt = prompt.replace("{{EVALUATION_RESULT}}", evaluation_str)

    # Get summary from Claude
    response = llm.invoke(prompt)
    response_content = response.content.strip()
    
    # Try to parse the response as JSON
    try:
        # First, try to extract JSON from the response if it's wrapped in markdown code blocks
        if "```json" in response_content:
            # Extract content between ```json and ```
            start_idx = response_content.find("```json") + 7
            end_idx = response_content.find("```", start_idx)
            if end_idx != -1:
                json_content = response_content[start_idx:end_idx].strip()
                parsed_json = json.loads(json_content)
                return json.dumps(parsed_json)
        
        # If no code blocks, try to parse the entire response as JSON
        parsed_json = json.loads(response_content)
        return json.dumps(parsed_json)
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return a fallback format
        print(f"Warning: Failed to parse JSON response for section {section_name}: {e}")
        print(f"Raw response: {response_content}")
        
        # Return a structured fallback
        return json.dumps({
            "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}",
            "Error": "JSON parsing failed - displaying raw response"
        })

async def generate_multiple_section_summaries(sections_data, llm, TESTING_MODE):
    """Generate summaries for multiple sections at once (5 sections per request)"""
    
    # non-testing mode
    async with aiofiles.open("prompt_summarize_by_section.txt", "r") as f:
        prompt_template = await f.read()

    # Process sections in batches of 5
    batch_size = 5
    section_summaries = {}
    
    # Get all section names
    section_names = list(sections_data.keys())
    
    for i in range(0, len(section_names), batch_size):
        batch_sections = section_names[i:i + batch_size]
        print(f"Processing batch of sections: {batch_sections}")
        
        # Prepare combined evaluation results for all sections in this batch
        combined_evaluation_results = {}
        
        for section_name in batch_sections:
            section_data = sections_data[section_name]
            
            # Check if there's no data for this section
            if not section_data:
                section_summaries[section_name] = json.dumps({
                    "Overall": f"""#### ⚠️ {section_name} – No Evaluation Data

Note: No evaluation data was provided for this section. This could indicate that:
- The section is missing from the model card
- No applicable policy requirements were found
- An error occurred during evaluation

Please ensure this section exists and contains the necessary information."""
                })
                continue
            
            # Format the evaluation results for this section
            evaluation_results = []
            for policy_name, policy_data in section_data.items():
                for article, score in policy_data['scores'].items():
                    if score < 5:  # Only include non-compliant items
                        description = policy_data['descriptions'].get(article, "No description available")
                        evaluation_results.append({
                            "policy": policy_name,
                            "article": article,
                            "score": score,
                            "description": description
                        })
            
            combined_evaluation_results[section_name] = evaluation_results
        
        # If no sections in this batch have evaluation results, handle them individually
        sections_with_data = [s for s in batch_sections if combined_evaluation_results.get(s)]
        if not sections_with_data:
            for section_name in batch_sections:
                if section_name not in section_summaries:  # Skip if already handled above
                    section_summaries[section_name] = json.dumps({
                        "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\nNo compliance issues were identified for this section. All evaluated criteria meet the requirements."
                    })
            continue
        
        # Format the combined evaluation results as a string
        combined_evaluation_str = json.dumps(combined_evaluation_results, indent=2)
        
        # Create a combined prompt for all sections in this batch
        # Use the original prompt template for each section, just combine them
        combined_prompt = ""
        for section_name in sections_with_data:
            # Use the original prompt template for each section
            section_prompt = prompt_template.replace("{{SECTION_NAME}}", section_name)
            section_prompt = section_prompt.replace("{{EVALUATION_RESULT}}", json.dumps(combined_evaluation_results[section_name], indent=2))
            combined_prompt += f"\n\n--- SECTION: {section_name} ---\n{section_prompt}"
        
        # Add instructions for the combined format
        combined_prompt = f"""Please process the following {len(sections_with_data)} sections. Each section has its own prompt below.

{combined_prompt}

Please provide a response for each section in the same format as the original prompts, separated by the section markers."""

        try:
            # Get summary from Claude
            response = llm.invoke(combined_prompt)
            response_content = response.content.strip()
            
            # Parse the response for each section
            for section_name in sections_with_data:
                try:
                    # Look for the section marker in the response - LLM is using "--- SECTION: Section Name ---" format
                    section_marker = f"--- SECTION: {section_name} ---"
                    if section_marker in response_content:
                        # Find the start of this section's response
                        start_idx = response_content.find(section_marker)
                        # Find the next section marker or end of response
                        next_section_idx = response_content.find("--- SECTION:", start_idx + len(section_marker))
                        
                        if next_section_idx != -1:
                            # Extract content between this section marker and the next one
                            section_response = response_content[start_idx + len(section_marker):next_section_idx].strip()
                        else:
                            # This is the last section, extract to the end
                            section_response = response_content[start_idx + len(section_marker):].strip()
                        
                        # Parse this section's response using the same logic as the original function
                        try:
                            # First, try to extract JSON from the response if it's wrapped in markdown code blocks
                            if "```json" in section_response:
                                # Extract content between ```json and ```
                                json_start = section_response.find("```json") + 7
                                json_end = section_response.find("```", json_start)
                                if json_end != -1:
                                    json_content = section_response[json_start:json_end].strip()
                                    parsed_json = json.loads(json_content)
                                    section_summaries[section_name] = json.dumps(parsed_json)
                                else:
                                    # If we can't find the closing ```, try to find the end of the JSON object
                                    # Look for the last closing brace
                                    brace_count = 0
                                    brace_end = -1
                                    for i in range(json_start, len(section_response)):
                                        if section_response[i] == '{':
                                            brace_count += 1
                                        elif section_response[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                brace_end = i + 1
                                                break
                                    
                                    if brace_end != -1:
                                        json_content = section_response[json_start:brace_end].strip()
                                        parsed_json = json.loads(json_content)
                                        section_summaries[section_name] = json.dumps(parsed_json)
                                    else:
                                        raise json.JSONDecodeError("Could not find end of JSON object", "", 0)
                            else:
                                # If no code blocks, try to parse the entire response as JSON
                                parsed_json = json.loads(section_response)
                                section_summaries[section_name] = json.dumps(parsed_json)
                                
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON response for section {section_name}: {e}")
                            print(f"Section response: {section_response}")
                            
                            # Return a structured fallback
                            section_summaries[section_name] = json.dumps({
                                "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\n{section_response}",
                                "Error": "JSON parsing failed - displaying raw response"
                            })
                    else:
                        # Fallback: try to find the section name in the response with different patterns
                        section_patterns = [
                            f"## SECTION: {section_name}",
                            f"## {section_name}",
                            f"### {section_name}",
                            f"**{section_name}**",
                            f'"{section_name}"'
                        ]
                        
                        found_pattern = None
                        for pattern in section_patterns:
                            if pattern in response_content:
                                found_pattern = pattern
                                break
                        
                        if found_pattern:
                            # Find the start of this section's content
                            start_idx = response_content.find(found_pattern)
                            
                            # Look for the JSON code block after this header
                            json_start = response_content.find("```json", start_idx)
                            if json_start != -1:
                                # Find the start of the actual JSON content
                                json_content_start = response_content.find("\n", json_start) + 1
                                # Find the end of the JSON code block
                                json_end = response_content.find("```", json_content_start)
                                
                                if json_end != -1:
                                    # Extract the JSON content
                                    json_content = response_content[json_content_start:json_end].strip()
                                    parsed_json = json.loads(json_content)
                                    section_summaries[section_name] = json.dumps(parsed_json)
                                else:
                                    # Try to find the end of the JSON object
                                    brace_count = 0
                                    brace_end = -1
                                    for i in range(json_content_start, len(response_content)):
                                        if response_content[i] == '{':
                                            brace_count += 1
                                        elif response_content[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                brace_end = i + 1
                                                break
                                    
                                    if brace_end != -1:
                                        json_content = response_content[json_content_start:brace_end].strip()
                                        parsed_json = json.loads(json_content)
                                        section_summaries[section_name] = json.dumps(parsed_json)
                                    else:
                                        raise json.JSONDecodeError("Could not find end of JSON object", "", 0)
                            else:
                                # If no JSON code block found, try to find JSON after the header
                                # Look for opening brace after the header
                                brace_start = response_content.find('{', start_idx)
                                if brace_start != -1:
                                    # Find the matching closing brace
                                    brace_count = 0
                                    brace_end = -1
                                    for i in range(brace_start, len(response_content)):
                                        if response_content[i] == '{':
                                            brace_count += 1
                                        elif response_content[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                brace_end = i + 1
                                                break
                                    
                                    if brace_end != -1:
                                        section_json = response_content[brace_start:brace_end]
                                        parsed_json = json.loads(section_json)
                                        section_summaries[section_name] = json.dumps(parsed_json)
                                    else:
                                        raise json.JSONDecodeError("Could not find matching closing brace", "", 0)
                                else:
                                    raise json.JSONDecodeError("Could not find JSON content for section", "", 0)
                        else:
                            raise json.JSONDecodeError(f"Section {section_name} not found in response", "", 0)
                            
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON response for section {section_name}: {e}")
                    print(f"Raw response: {response_content}")
                    
                    # Return a structured fallback
                    section_summaries[section_name] = json.dumps({
                        "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\nUnable to parse response for this section.",
                        "Error": "JSON parsing failed - displaying raw response"
                    })
                    
        except Exception as e:
            print(f"Error processing batch of sections {batch_sections}: {str(e)}")
            # Handle each section individually as fallback
            for section_name in batch_sections:
                if section_name not in section_summaries:
                    section_summaries[section_name] = json.dumps({
                        "Overall": f"#### ❌ {section_name} – Error\n\nAn error occurred while generating the summary for this section. Please check the logs for more details.",
                        "Error": str(e)
                    })
    
    return section_summaries