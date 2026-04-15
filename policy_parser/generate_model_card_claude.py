import os
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0.3
)

def extract_text_from_md(folder_path):
    """Reads Markdown template parts and returns combined content in order."""
    md_parts = sorted([f for f in os.listdir(folder_path) if f.endswith(".md")])

    combined_text = {}
    for md_file in md_parts:
        with open(os.path.join(folder_path, md_file), "r", encoding="utf-8") as f:
            combined_text[md_file] = f.read().strip()

    return combined_text

md_template_folder = "mc_template"
template_parts = extract_text_from_md(md_template_folder)


# Markdown table prompt with structured output format
prompt_template = PromptTemplate(
    input_variables=["system_name", "template_text"],
    template="""
    You are an AI governance researcher. Use the following Model Card template as a strict format guide to generate example model cards for certain AI systems.

    **Model Card Template:**
    {template_text}

    ##Instructions:
    - Generate a new model card in **Markdown table format** for a given AI system.
    - DO NOT modify/miss out/add extra section names from the template. MUST match exactly with the template format.
    - Fill in the content column for each section.
    - Don't include column "description" in output. It is a short description for guidance. 
    - Don't include column "example write-up" in output. It is an example to show you the length, depth and format for output. 
    - Bullet points are encouraged for longer, data-driven content as demonstrated in the column "example write-up" using a markdown friendly approach - do not change line, just make sure to number the points and put one point after another, e.g. (1) apple; (2) pear; (3) banana. 
    - Include some simple detail for explaination and depth, E.g. do "(1) Privacy Concerns: Customers may not be aware they are being tracked, raising ethical and legal issues; (2) Demographic Bias: The system may misinterpret behaviors of underrepresented groups, such as people with disabilities or those wearing cultural attire." instead of "(1) Privacy Concerns; (2) Demographic Bias"
    - Example output:

    ```
    | Section | Content |
    | ------- | ------- |
    | System Name | AI-powered Retail Analytics System |
    ```

    - Return only the formatted markdown table.
    - No additional explanations, just the table itself.

    Now, generate an example Model Card for an AI system using the structure above:

    - **System Name:** {system_name}

    """
)

def generate_model_card(system_name):
    generated_content = []
    
    for i, (part_name, template_text) in enumerate(template_parts.items()):
        print(f"📄 Processing {part_name} for {system_name}...")
        
        prompt = prompt_template.format(system_name=system_name, template_text=template_text)
        response = llm.invoke(prompt).content

        if i == 1:  # Ensure Part2 doesn't repeat the table header
            response = "\n".join(response.split("\n")[2:])
        
        generated_content.append(response)

    return "\n".join(generated_content)


def save_model_card_as_markdown(system_name, model_card_content, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_base_name = os.path.join(output_folder, f"{system_name.replace(' ', '_').lower()}_model_card")
    
    md_file = f"{file_base_name}.md"
    txt_file = f"{file_base_name}.txt"

    # Save markdown file
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    # Save plain text file (same markdown format)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(model_card_content)

    print(f"✅ Model Card saved: {md_file} and {txt_file}")

example_systems = [
    {"system_name": "AI Resume Screening Tool"},
    {"system_name": "Autonomous Driving Decision Engine"},
    {"system_name": "AI-Powered Insurance Claim Evaluator"}
]

output_folder = "model_card_outputs"

for system in example_systems:
    print(f"📄 Generating Model Card for {system['system_name']}...\n")
    model_card_content = generate_model_card(system["system_name"])
    print(model_card_content, "\n")
    save_model_card_as_markdown(system["system_name"], model_card_content, output_folder)

print("✅ All Model Cards have been processed and saved.")
