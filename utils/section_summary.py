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
        # First, check if the response is plain text (for fully compliant sections)
        if response_content.strip().lower() == "this section is fully compliant and no actions needed.":
            return json.dumps({
                "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\nThis section is fully compliant and no actions needed."
            })
        
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
        try:
            parsed_json = json.loads(response_content)
            return json.dumps(parsed_json)
        except json.JSONDecodeError:
            # If JSON parsing fails, check if it's a plain text response
            if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
                return json.dumps({
                    "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\n{response_content}"
                })
            else:
                # Return a structured fallback for other plain text responses
                return json.dumps({
                    "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}",
                    "Error": "JSON parsing failed - displaying raw response"
                })
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return a fallback format
        print(f"Warning: Failed to parse JSON response for section {section_name}: {e}")
        print(f"Raw response: {response_content}")
        
        # Check if it's a plain text response for fully compliant sections
        if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
            return json.dumps({
                "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\n{response_content}"
            })
        else:
            # Return a structured fallback
            return json.dumps({
                "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}",
                "Error": "JSON parsing failed - displaying raw response"
            })

async def generate_multiple_section_summaries(sections_data, llm, TESTING_MODE):
    """Generate summaries for multiple sections at once (5 sections per request)"""
    
    # Load the multi-section prompt template
    async with aiofiles.open("prompt_summarize_by_section.txt", "r") as f:
        multi_section_prompt = await f.read()

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
        
        # Create the sections input for the prompt
        sections_input = ""
        for section_name in sections_with_data:
            sections_input += f"Section name: {section_name}\nEvaluation Result: {json.dumps(combined_evaluation_results[section_name], indent=2)}\n\n"
        
        # Prepare the prompt
        prompt = multi_section_prompt.replace("{{SECTIONS_INPUT}}", sections_input)

        try:
            # Get summary from Claude
            response = llm.invoke(prompt)
            response_content = response.content.strip()
            
            # Parse the response - it should be a JSON object with section names as keys
            try:
                # First, try to extract JSON from the response if it's wrapped in markdown code blocks
                if "```json" in response_content:
                    # Extract content between ```json and ```
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    if json_end != -1:
                        json_content = response_content[json_start:json_end].strip()
                        parsed_response = json.loads(json_content)
                    else:
                        # If we can't find the closing ```, try to find the end of the JSON object
                        brace_count = 0
                        brace_end = -1
                        for i in range(json_start, len(response_content)):
                            if response_content[i] == '{':
                                brace_count += 1
                            elif response_content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i + 1
                                    break
                        
                        if brace_end != -1:
                            json_content = response_content[json_start:brace_end].strip()
                            parsed_response = json.loads(json_content)
                        else:
                            raise json.JSONDecodeError("Could not find end of JSON object", "", 0)
                else:
                    # If no code blocks, try to parse the entire response as JSON
                    parsed_response = json.loads(response_content)
                
                # Process each section's response
                for section_name in sections_with_data:
                    if section_name in parsed_response:
                        section_summaries[section_name] = json.dumps(parsed_response[section_name])
                    else:
                        # If section not found in response, check if it's a plain text response
                        if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
                            section_summaries[section_name] = json.dumps({
                                "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\nThis section is fully compliant and no actions needed."
                            })
                        else:
                            section_summaries[section_name] = json.dumps({
                                "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\nSection response not found in batch response.",
                                "Error": "Section not found in batch response"
                            })
                            
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON response for batch {batch_sections}: {e}")
                print(f"Raw response: {response_content}")
                
                # Fallback: try to parse each section individually
                for section_name in sections_with_data:
                    if section_name not in section_summaries:
                        # Check if there's a plain text response for this section
                        if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
                            section_summaries[section_name] = json.dumps({
                                "Overall": f"#### 🟢 {section_name} – Fully Compliant\n\nThis section is fully compliant and no actions needed."
                            })
                        else:
                            section_summaries[section_name] = json.dumps({
                                "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\nUnable to parse batch response for this section.",
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