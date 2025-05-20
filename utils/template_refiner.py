import os
import json
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import re

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

class TemplateRefiner:
    def __init__(self, template_dir: str, policy_dir: str, api_key: str):
        self.template_dir = Path(template_dir)
        self.policy_dir = Path(policy_dir)
        self.llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.3
        )
        # Define a consistent AI system example that will be used across all sections
        self.system_example = {
            "name": "RetailVisionAI",
            "type": "AI-powered customer tracking system",
            "purpose": "Analyze video streams from security cameras for customer traffic, dwell time, and queue analytics",
            "key_features": [
                "Computer vision models for customer tracking",
                "Edge computing for on-premise processing",
                "Cloud-based APIs (Azure AI Vision)",
                "Real-time heatmaps and analytics dashboard"
            ]
        }
        
    def read_template_files(self) -> Dict[str, str]:
        """Read all template files and combine their content."""
        template_content = {}
        for file in self.template_dir.glob("*.md"):
            with open(file, 'r', encoding='utf-8') as f:
                template_content[file.name] = f.read()
        return template_content

    def read_policy_files(self) -> Dict[str, str]:
        """Read all policy files and combine their content."""
        policy_content = {}
        for file in self.policy_dir.glob("*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                policy_content[file.name] = f.read()
        return policy_content

    def parse_markdown_table(self, content: str) -> List[Dict]:
        """Parse markdown table format into sections."""
        sections = []
        lines = content.strip().split('\n')
        
        # Skip header and separator lines
        data_lines = [line for line in lines if line.startswith('|') and not line.startswith('| :')]
        
        for line in data_lines:
            # Split by | and remove empty cells
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) >= 2:  # At least title and description
                sections.append({
                    'title': cells[0],
                    'description': cells[1],
                    'example': cells[2] if len(cells) > 2 else ''
                })
        
        return sections

    def refine_section(self, section: Dict, policy_content: Dict[str, str], all_sections: List[Dict]) -> None:
        """Use Claude to refine a section based on policy requirements."""
        # Create context about the system for consistent examples
        system_context = f"""
        We are documenting an AI system with the following characteristics:
        - Name: {self.system_example['name']}
        - Type: {self.system_example['type']}
        - Purpose: {self.system_example['purpose']}
        - Key Features: {', '.join(self.system_example['key_features'])}
        """

        prompt = f"""You are an expert in AI policy compliance and model card documentation. 
        Your task is to analyze and improve the following model card section to ensure it captures all necessary information for policy compliance assessment.

        {system_context}

        Current Section:
        Title: {section['title']}
        Description: {section['description']}
        Current Example: {section['example']}

        Policy Requirements:
        {json.dumps(policy_content, indent=2)}

        Please provide:
        1. An improved description that ensures all necessary information for policy compliance is captured (not specific just to the provided section, think of it as a general improvement to the field description to better guide developers in filling in the model card). Also don't change the title. Just try to improve the description which guides user on what information to provide in each field.
        2. A comprehensive example that demonstrates compliance with relevant policies, maintaining consistency with the system described above
        3. An explanation of why each modification was made, referencing specific policy articles

        Please format your response with clear sections:
        1. Improved Description:
        [Your improved description here]

        2. Improved Example:
        [Your improved example here]

        3. Modification Explanations:
        - Policy Article [X]: [Explanation of why this change was needed]
        - Policy Article [Y]: [Explanation of why this change was needed]
        etc.
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert in AI policy compliance and model card documentation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.llm.invoke(messages)
        print(response.content)
        print("\n" + "="*80 + "\n")

    def refine_templates(self) -> None:
        """Refine all templates based on policy requirements."""
        template_content = self.read_template_files()
        policy_content = self.read_policy_files()
        
        # Combine all sections from both template files
        all_sections = []
        for content in template_content.values():
            sections = self.parse_markdown_table(content)
            all_sections.extend(sections)

        # Process each section
        for section in all_sections:
            print(f"\n{'='*80}")
            print(f"Processing Section: {section['title']}")
            print(f"{'='*80}\n")
            
            self.refine_section(section, policy_content, all_sections)

def main():
    # Get API key from environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Initialize the refiner
    refiner = TemplateRefiner(
        template_dir="mc_template",
        policy_dir="policies",
        api_key=api_key
    )

    # Refine templates and print results
    refiner.refine_templates()

if __name__ == "__main__":
    main() 