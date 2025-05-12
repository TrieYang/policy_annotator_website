import os
from dotenv import load_dotenv
import aiofiles

async def generate_top_level_summary(policy_summaries, llm):
    """Generate a top-level summary across all policy evaluations"""
    try:
        # Read the prompt template
        async with aiofiles.open("prompt_top_level_summary.txt", "r") as f:
            prompt_template = await f.read()

        # Format the summaries for the prompt
        summaries_json = json.dumps({"policy_summaries": policy_summaries}, indent=2)
        
        # Prepare the prompt
        prompt = prompt_template.replace("{{POLICY_SUMMARIES}}", summaries_json)

        # Get summary from Claude
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating top-level summary: {str(e)}")
        return None