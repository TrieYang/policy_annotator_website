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