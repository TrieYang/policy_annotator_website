import os
import time
import json
import asyncio
from dotenv import load_dotenv
import aiofiles

async def generate_section_summary(section_name, section_data, llm, TESTING_MODE):
    """Generate a summary of compliance evaluation results for a specific model card section"""
    
    # Add timestamp to show when this task actually started
    start_timestamp = time.strftime("%H:%M:%S")
    print(f"[{start_timestamp}] 🚀 TASK STARTED for section '{section_name}'")
    
    try:
        # Load the prompt template
        async with aiofiles.open("prompt_summarize_by_section.txt", "r") as f:
            prompt_template = await f.read()

        # Format the evaluation results for the prompt
        evaluation_results = []
        all_policies = set()
        
        # First, collect all policy names and check for non-compliant items
        for policy_name, policy_data in section_data.items():
            all_policies.add(policy_name)
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
            print(f"Section '{section_name}' has no compliance issues - marking as fully compliant")
            # Return JSON with all policies marked as compliant
            result_json = {"Overall": "This section is fully compliant and no actions needed."}
            for policy_name in all_policies:
                result_json[policy_name] = "This section is fully compliant and no actions needed."
            return json.dumps(result_json)

        # Format the evaluation results as a string
        evaluation_str = json.dumps(evaluation_results, indent=2)
        print(f"Section '{section_name}' has {len(evaluation_results)} compliance issues to process")

        # Prepare the prompt
        prompt = prompt_template.replace("{{SECTIONS_INPUT}}", f"Section name: {section_name}\nEvaluation Result: {evaluation_str}")

        # Get summary from Claude
        print(f"Requesting summary for section '{section_name}' from LLM...")
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        print(f"Received response for section '{section_name}' (length: {len(response_content)} characters)")
        
        # Special debugging for problematic sections
        if section_name == "Contact Info":
            print(f"DEBUG - Contact Info raw response: '{response_content}'")
            print(f"DEBUG - Response type: {type(response_content)}")
            print(f"DEBUG - Response repr: {repr(response_content)}")
            print(f"DEBUG - Response starts with: {repr(response_content[:20])}")
            print(f"DEBUG - Response ends with: {repr(response_content[-20:])}")
            print(f"DEBUG - Contains curly braces: {{ in response: {'{' in response_content}, }} in response: {'}' in response_content}")
            print(f"DEBUG - Contains brackets: [ in response: {'[' in response_content}, ] in response: {']' in response_content}")
        
        # Check for potentially truncated responses
        if len(response_content) < 50:
            print(f"Warning: Response for section '{section_name}' seems too short, may be truncated")
            result_json = {
                "Overall": f"#### ⚠️ {section_name} – Response Too Short\n\nThe response received was too short and may be truncated. Please check the logs for details.",
                "Error": "Response too short - may be truncated"
            }
            # Add all policies with error message
            for policy_name in all_policies:
                result_json[policy_name] = f"#### ⚠️ {section_name} – Response Too Short\n\nThe response received was too short and may be truncated. Please check the logs for details."
            return json.dumps(result_json)
        
        # Check if response looks like it might be a simple string (not JSON)
        if not any(char in response_content for char in ['{', '[', '"']):
            print(f"Warning: Response for section '{section_name}' doesn't look like JSON, treating as plain text")
            if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
                result_json = {"Overall": "This section is fully compliant and no actions needed."}
                # Add all policies as fully compliant
                for policy_name in all_policies:
                    result_json[policy_name] = "This section is fully compliant and no actions needed."
                return json.dumps(result_json)
            else:
                result_json = {
                    "Overall": f"#### ⚠️ {section_name} – Plain Text Response\n\n{response_content}",
                    "Error": "Response appears to be plain text, not JSON"
                }
                # Add all policies with error message
                for policy_name in all_policies:
                    result_json[policy_name] = f"#### ⚠️ {section_name} – Plain Text Response\n\n{response_content}"
                return json.dumps(result_json)
        
        # Check if response is a quoted string that represents fully compliant
        if response_content.startswith('"') and response_content.endswith('"'):
            try:
                # Try to parse as JSON string
                parsed_string = json.loads(response_content)
                if "fully compliant" in parsed_string.lower() or "no actions needed" in parsed_string.lower():
                    print(f"Section '{section_name}' marked as fully compliant via quoted string (legacy format)")
                    
                    # Check for potential mismatch between evaluation data and LLM response
                    if evaluation_results:
                        print(f"WARNING: Section '{section_name}' has {len(evaluation_results)} compliance issues but LLM marked it as fully compliant!")
                        print(f"WARNING: This suggests a potential issue with the prompt or evaluation data")
                    
                    result_json = {"Overall": "This section is fully compliant and no actions needed."}
                    # Add all policies as fully compliant
                    for policy_name in all_policies:
                        result_json[policy_name] = "This section is fully compliant and no actions needed."
                    return json.dumps(result_json)
            except json.JSONDecodeError:
                pass  # Not a valid JSON string, continue with normal parsing
        
        # Check for common LLM response issues
        if response_content.count('{') != response_content.count('}'):
            print(f"Warning: Response for section '{section_name}' has mismatched braces, may be incomplete")
            result_json = {
                "Overall": f"#### ⚠️ {section_name} – Incomplete Response\n\nThe response appears to be incomplete (mismatched braces). Please check the logs for details.",
                "Error": "Incomplete response - mismatched braces"
            }
            # Add all policies with error message
            for policy_name in all_policies:
                result_json[policy_name] = f"#### ⚠️ {section_name} – Incomplete Response\n\nThe response appears to be incomplete (mismatched braces). Please check the logs for details."
            return json.dumps(result_json)
        
        # Try to parse the response as JSON
        try:
            # First, check if the response is plain text (for fully compliant sections)
            if response_content.strip().lower() == "this section is fully compliant and no actions needed.":
                print(f"Section '{section_name}' marked as fully compliant by LLM")
                result_json = {"Overall": "This section is fully compliant and no actions needed."}
                # Add all policies as fully compliant
                for policy_name in all_policies:
                    result_json[policy_name] = "This section is fully compliant and no actions needed."
                return json.dumps(result_json)
            
            # First, try to extract JSON from the response if it's wrapped in markdown code blocks
            if "```json" in response_content:
                # Extract content between ```json and ```
                start_idx = response_content.find("```json") + 7
                end_idx = response_content.find("```", start_idx)
                if end_idx != -1:
                    json_content = response_content[start_idx:end_idx].strip()
                    parsed_json = json.loads(json_content)
                    print(f"Successfully parsed JSON from code block for section '{section_name}'")
                    
                    # Validate the parsed JSON structure
                    if not isinstance(parsed_json, dict):
                        print(f"ERROR: Parsed JSON for section '{section_name}' is not a dictionary")
                        print(f"ERROR: Type: {type(parsed_json)}")
                        print(f"ERROR: Value: {parsed_json}")
                        print(f"ERROR: Raw response was: {repr(response_content)}")
                        
                        # Special handling for lists - convert to dict if possible
                        if isinstance(parsed_json, list):
                            print(f"INFO: Converting list response to dictionary for section '{section_name}'")
                            if len(parsed_json) == 1 and isinstance(parsed_json[0], dict):
                                parsed_json = parsed_json[0]
                                print(f"INFO: Successfully converted list with single dict to dict")
                            else:
                                # Convert list to dict with Overall key
                                parsed_json = {"Overall": f"#### ⚠️ {section_name} – List Response\n\nReceived list response: {parsed_json}"}
                                print(f"INFO: Converted list to dict with Overall key")
                        else:
                            raise ValueError(f"Parsed JSON is not a dictionary, got {type(parsed_json)}: {parsed_json}")
                    
                    # Ensure it has at least an "Overall" key
                    if "Overall" not in parsed_json:
                        print(f"Warning: Parsed JSON for section '{section_name}' missing 'Overall' key, adding fallback")
                        parsed_json["Overall"] = f"#### ⚠️ {section_name} – Evaluation Summary\n\nResponse missing 'Overall' summary."
                    
                    # Validate that all values are strings
                    for key, value in parsed_json.items():
                        if not isinstance(value, str):
                            print(f"Warning: Non-string value found for key '{key}' in section '{section_name}', converting to string")
                            parsed_json[key] = str(value)
                    
                    # Ensure all policies are included in the result
                    for policy_name in all_policies:
                        if policy_name not in parsed_json:
                            # If policy not in LLM response, mark as fully compliant
                            parsed_json[policy_name] = "This section is fully compliant and no actions needed."
                    
                    return json.dumps(parsed_json)
                else:
                    print(f"Warning: Found ```json but no closing ``` for section '{section_name}'")
            
            # If no code blocks, try to parse the entire response as JSON
            try:
                parsed_json = json.loads(response_content)
                print(f"Successfully parsed JSON response for section '{section_name}'")
                
                # Check if the parsed JSON is a quoted string (fully compliant response)
                if isinstance(parsed_json, str):
                    if "fully compliant" in parsed_json.lower() or "no actions needed" in parsed_json.lower():
                        print(f"Section '{section_name}' marked as fully compliant via quoted string")
                        
                        # Check for potential mismatch between evaluation data and LLM response
                        if evaluation_results:
                            print(f"WARNING: Section '{section_name}' has {len(evaluation_results)} compliance issues but LLM marked it as fully compliant!")
                            print(f"WARNING: This suggests a potential issue with the prompt or evaluation data")
                        
                        result_json = {"Overall": "This section is fully compliant and no actions needed."}
                        # Add all policies as fully compliant
                        for policy_name in all_policies:
                            result_json[policy_name] = "This section is fully compliant and no actions needed."
                        return json.dumps(result_json)
                    else:
                        print(f"Warning: Parsed JSON is a string but not 'fully compliant': {parsed_json}")
                        result_json = {
                            "Overall": f"#### ⚠️ {section_name} – String Response\n\n{parsed_json}",
                            "Error": "LLM returned string response instead of JSON object"
                        }
                        # Add all policies with error message
                        for policy_name in all_policies:
                            result_json[policy_name] = f"#### ⚠️ {section_name} – String Response\n\n{parsed_json}"
                        return json.dumps(result_json)
                
                # Validate the parsed JSON structure
                if not isinstance(parsed_json, dict):
                    print(f"ERROR: Parsed JSON for section '{section_name}' is not a dictionary")
                    print(f"ERROR: Type: {type(parsed_json)}")
                    print(f"ERROR: Value: {parsed_json}")
                    print(f"ERROR: Raw response was: {repr(response_content)}")
                    
                    # Special handling for lists - convert to dict if possible
                    if isinstance(parsed_json, list):
                        print(f"INFO: Converting list response to dictionary for section '{section_name}'")
                        if len(parsed_json) == 1 and isinstance(parsed_json[0], dict):
                            parsed_json = parsed_json[0]
                            print(f"INFO: Successfully converted list with single dict to dict")
                        else:
                            # Convert list to dict with Overall key
                            parsed_json = {"Overall": f"#### ⚠️ {section_name} – List Response\n\nReceived list response: {parsed_json}"}
                            print(f"INFO: Converted list to dict with Overall key")
                    else:
                        raise ValueError(f"Parsed JSON is not a dictionary, got {type(parsed_json)}: {parsed_json}")
                
                # Ensure it has at least an "Overall" key
                if "Overall" not in parsed_json:
                    print(f"Warning: Parsed JSON for section '{section_name}' missing 'Overall' key, adding fallback")
                    parsed_json["Overall"] = f"#### ⚠️ {section_name} – Evaluation Summary\n\nResponse missing 'Overall' summary."
                
                # Validate that all values are strings
                for key, value in parsed_json.items():
                    if not isinstance(value, str):
                        print(f"Warning: Non-string value found for key '{key}' in section '{section_name}', converting to string")
                        parsed_json[key] = str(value)
                
                # Ensure all policies are included in the result
                for policy_name in all_policies:
                    if policy_name not in parsed_json:
                        # If policy not in LLM response, mark as fully compliant
                        parsed_json[policy_name] = "This section is fully compliant and no actions needed."
                
                return json.dumps(parsed_json)
            except json.JSONDecodeError:
                # If JSON parsing fails, check if it's a plain text response
                if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
                    print(f"Section '{section_name}' appears to be fully compliant based on text content")
                    result_json = {"Overall": "This section is fully compliant and no actions needed."}
                    # Add all policies as fully compliant
                    for policy_name in all_policies:
                        result_json[policy_name] = "This section is fully compliant and no actions needed."
                    return json.dumps(result_json)
                else:
                    # Return a structured fallback for other plain text responses
                    print(f"Warning: JSON parsing failed for section '{section_name}', returning fallback format")
                    result_json = {
                        "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}",
                        "Error": "JSON parsing failed - displaying raw response"
                    }
                    # Add all policies with error message
                    for policy_name in all_policies:
                        result_json[policy_name] = f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}"
                    return json.dumps(result_json)
            
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, return a fallback format
            print(f"Warning: Failed to parse JSON response for section {section_name}: {e}")
            print(f"Raw response: {response_content}")
            
            # Check if it's a plain text response for fully compliant sections
            if "fully compliant" in response_content.lower() or "no actions needed" in response_content.lower():
                print(f"Section '{section_name}' appears to be fully compliant based on text content")
                result_json = {"Overall": "This section is fully compliant and no actions needed."}
                # Add all policies as fully compliant
                for policy_name in all_policies:
                    result_json[policy_name] = "This section is fully compliant and no actions needed."
                return json.dumps(result_json)
            else:
                # Return a structured fallback
                print(f"Section '{section_name}' - returning error fallback due to JSON parsing failure")
                result_json = {
                    "Overall": f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}",
                    "Error": f"JSON parsing failed: {str(e)} - displaying raw response"
                }
                # Add all policies with error message
                for policy_name in all_policies:
                    result_json[policy_name] = f"#### ⚠️ {section_name} – Evaluation Summary\n\n{response_content}"
                return json.dumps(result_json)
                
    except Exception as e:
        print(f"Unexpected error in generate_section_summary for section '{section_name}': {str(e)}")
        # Get all policies from section_data for error case
        all_policies = set(section_data.keys()) if section_data else set()
        result_json = {
            "Overall": f"#### ❌ {section_name} – Error\n\nAn unexpected error occurred while generating the summary for this section: {str(e)}",
            "Error": str(e)
        }
        # Add all policies with error message
        for policy_name in all_policies:
            result_json[policy_name] = f"#### ❌ {section_name} – Error\n\nAn unexpected error occurred while generating the summary for this section: {str(e)}"
        return json.dumps(result_json)

async def generate_multiple_section_summaries(sections_data, llm, TESTING_MODE):
    """Generate summaries for multiple sections by processing all sections concurrently"""
    
    section_summaries = {}
    
    # Get all unique policy names across all sections
    all_policies = set()
    for section_name, section_data in sections_data.items():
        if section_data:
            all_policies.update(section_data.keys())
    
    # Create a list to store all concurrent tasks
    all_tasks = []
    task_metadata = {}  # To track which task corresponds to which section
    
    print(f"\n{'='*50}")
    print(f"Preparing {len(sections_data)} sections for concurrent summary generation...")
    print(f"{'='*50}")
    
    # Prepare all tasks first without executing them
    for section_name in sections_data.keys():
        print(f"Preparing section: {section_name}")
        
        section_data = sections_data[section_name]
        
        # Check if there's no data for this section
        if not section_data:
            print(f"Section '{section_name}' has no evaluation data - will create fallback result")
            # Create a task that returns immediately with fallback result
            async def create_fallback_result(section_name, all_policies):
                result_json = {
                    "Overall": f"""#### ⚠️ {section_name} – No Evaluation Data

Note: No evaluation data was provided for this section. This could indicate that:
- The section is missing from the model card
- No applicable policy requirements were found
- An error occurred during evaluation

Please ensure this section exists and contains the necessary information."""
                }
                # Add all policies with the same message
                for policy_name in all_policies:
                    result_json[policy_name] = f"""#### ⚠️ {section_name} – No Evaluation Data

Note: No evaluation data was provided for this section. This could indicate that:
- The section is missing from the model card
- No applicable policy requirements were found
- An error occurred during evaluation

Please ensure this section exists and contains the necessary information."""
                
                return section_name, json.dumps(result_json)
            
            task = asyncio.create_task(create_fallback_result(section_name, all_policies))
            task_metadata[len(all_tasks)] = {'section_name': section_name, 'type': 'fallback'}
            all_tasks.append(task)
            continue
        
        # Create task for actual summary generation
        task = asyncio.create_task(generate_section_summary(section_name, section_data, llm, TESTING_MODE))
        task_metadata[len(all_tasks)] = {'section_name': section_name, 'type': 'summary'}
        all_tasks.append(task)
    
    print(f"\n{'='*50}")
    print(f"Executing {len(all_tasks)} section summary requests concurrently...")
    print(f"{'='*50}")
    print(f"ALL TASKS STARTED AT THE SAME TIME - they are running in parallel!")
    print(f"Watch the timestamps below - they should be very close together!")
    print(f"{'='*50}")
    
    # Execute all tasks concurrently
    start_time = time.time()
    responses = await asyncio.gather(*all_tasks, return_exceptions=True)
    end_time = time.time()
    
    print(f"Completed {len(all_tasks)} section summary requests in {end_time - start_time:.2f} seconds")
    print(f"Average time per request: {(end_time - start_time) / len(all_tasks):.2f} seconds")
    
    # Process all responses
    print(f"\n{'='*50}")
    print(f"Processing {len(responses)} section summary responses...")
    print(f"{'='*50}")
    
    successful_requests = 0
    failed_requests = 0
    
    for task_id, response in enumerate(responses):
        metadata = task_metadata[task_id]
        section_name = metadata['section_name']
        task_type = metadata['type']
        
        if isinstance(response, Exception):
            print(f"Request failed for section '{section_name}' ({task_type}): {str(response)}")
            failed_requests += 1
            
            # Create error fallback result
            result_json = {
                "Overall": f"#### ❌ {section_name} – Error\n\nAn error occurred while generating the summary for this section. Please check the logs for more details.",
                "Error": str(response)
            }
            # Add all policies with the same error message
            for policy_name in all_policies:
                result_json[policy_name] = f"#### ❌ {section_name} – Error\n\nAn error occurred while generating the summary for this section. Please check the logs for more details."
            
            section_summaries[section_name] = json.dumps(result_json)
            continue
        
        try:
            if task_type == 'fallback':
                # This was a fallback task that returned (section_name, result)
                section_name, result = response
                section_summaries[section_name] = result
                print(f"Processed fallback result for section: {section_name}")
            else:
                # This was a summary generation task
                section_summaries[section_name] = response
                print(f"Processed summary response for section: {section_name}")
            
            successful_requests += 1
            
        except Exception as e:
            print(f"Error processing response for section '{section_name}' ({task_type}): {str(e)}")
            failed_requests += 1
            
            # Create error fallback result
            result_json = {
                "Overall": f"#### ❌ {section_name} – Error\n\nAn error occurred while processing the summary for this section: {str(e)}",
                "Error": str(e)
            }
            # Add all policies with the same error message
            for policy_name in all_policies:
                result_json[policy_name] = f"#### ❌ {section_name} – Error\n\nAn error occurred while processing the summary for this section: {str(e)}"
            
            section_summaries[section_name] = json.dumps(result_json)
    
    print(f"\n{'='*50}")
    print(f"Section Summary Processing Summary:")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Total requests: {len(all_tasks)}")
    print(f"{'='*50}")
    
    return section_summaries