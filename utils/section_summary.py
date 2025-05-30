import os
import time
import json
from dotenv import load_dotenv
import aiofiles

async def generate_section_summary(section_name, section_data, llm, TESTING_MODE):
    """Generate a summary of compliance evaluation results for a specific model card section"""
    if TESTING_MODE:
        # Sample responses for testing mode
        sample_responses = {
                "System Name": """{
            "Overall": "#### Issues and Fixes:\\n- **Ambiguous system name**  \\n↳ *Clarify the system name to avoid confusion with other tools or versions.*\\n- **Missing version identifier**  \\n↳ *Include version number in the system name for better traceability.*\\n- **Inconsistent naming across documentation**  \\n↳ *Standardize system name usage across all technical and user documentation.*\\n- **No clear distinction from similar systems**  \\n↳ *Add unique identifiers to differentiate from related AI systems.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Missing unique identifier**  \\n↳ *Include a distinct and traceable system name or code to support regulatory filing.*\\n- **No conformity assessment reference**  \\n↳ *Link system name to EU conformity assessment documentation.*",
            "AIDA": "#### Issues and Fixes:\\n- **Inconsistent system labeling**  \\n↳ *Ensure the system name is uniform across all public and internal documentation.*\\n- **Missing risk classification indicator**  \\n↳ *Include risk level designation in system naming convention.*",
            "CCPA": "#### Issues and Fixes:\\n- **Name not tied to consumer-facing functionality**  \\n↳ *Clearly indicate which services use this AI system so consumers understand its presence.*\\n- **No privacy notice reference**  \\n↳ *Link system name to relevant privacy notices and data handling policies.*"
            CCPA": "#### Issues and Fixes:\\n- **Versioning not tied to data retention**  \\n↳ *Clarify whether updates reset or affect data collection timelines.*\\n- **Missing privacy updates**  \\n↳ *Document updates affecting privacy.*"
            }"""
            }

        
        return sample_responses.get(section_name, f"""#### ⚠️ {section_name} – No Evaluation Data
Note: No evaluation data was provided for this section.""")
    
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
        return f"""#### 🟢 {section_name} – Fully Compliant
        No compliance issues were identified for this section. All evaluated criteria meet the requirements."""

    # Format the evaluation results as a string
    evaluation_str = json.dumps(evaluation_results, indent=2)

    # Prepare the prompt
    prompt = prompt_template.replace("{{SECTION_NAME}}", section_name)
    prompt = prompt.replace("{{EVALUATION_RESULT}}", evaluation_str)

    # Get summary from Claude
    response = llm.invoke(prompt)
    return response.content.strip()