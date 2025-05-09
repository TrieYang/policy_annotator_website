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

SAMPLE_RESPONSE_FOLDER = "./sample_responses"
TESTING_MODE = True

class FakeLLM:
    def __init__(self, sample_folder):
        self.sample_folder = sample_folder
        self.current_policy = None
        self.current_section = None

    def set_context(self, policy, section):
        self.current_policy = policy
        self.current_section = section

    def _section_to_filename(self, section):
        # Convert section name to filename format
        # e.g., "System Name" -> "System_Name"
        # e.g., "Primary Developer/Org" -> "Primary_Developer_Org"
        # e.g., "Out-of-scope use cases" -> "Out_of_scope_Use_Cases"
        # First replace special characters with underscores
        filename = section.replace(" ", "_").replace("/", "_").replace("-", "_")
        # Then capitalize each word
        words = filename.split("_")
        capitalized_words = [word.capitalize() for word in words]
        return "_".join(capitalized_words)

    def invoke(self, messages):
        if not self.current_policy or not self.current_section:
            raise ValueError("Policy and section must be set before invoking FakeLLM")

        filename = self._section_to_filename(self.current_section)
        sample_file_path = os.path.join(self.sample_folder, self.current_policy, f"{filename}.md")
        
        if not os.path.exists(sample_file_path):
            raise FileNotFoundError(f"Sample response not found: {sample_file_path}")

        with open(sample_file_path, "r") as f:
            content = f.read()

        return type("Response", (object,), {
            "content": content,
            "usage_metadata": {"input_token_details": "mocked"}
        })()

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")



fakeLlm = FakeLLM(SAMPLE_RESPONSE_FOLDER)

llm = ChatAnthropic(
model="claude-3-5-sonnet-20241022",
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

def parse_model_card_content(model_card_content):
    """Parse model card content into sections"""
    sections_content = {}
    current_section = None
    current_content = []
    
    for line in model_card_content.split('\n'):
        if line.startswith('|') and '|' in line[1:]:
            # This is a table row
            if current_section:
                current_content.append(line)
        elif line.strip():
            # This is a section header
            if current_section and current_content:
                sections_content[current_section] = '\n'.join(current_content)
            current_section = line.strip()
            current_content = []
    
    # Add the last section
    if current_section and current_content:
        sections_content[current_section] = '\n'.join(current_content)
    
    return sections_content

def generate_interactive_heatmap(data_df, descriptions_df, policy, model_card_content=None, output_filename=None):
    """Generate an interactive heatmap using Plotly"""
    import plotly.graph_objects as go
    import time
    import pandas as pd
    
    # Create a unique filename if none provided
    if output_filename is None:
        timestamp = int(time.time())
        output_filename = f"heatmap_{timestamp}.html"

    # Get unique policy names and their positions
    policy_positions = {}
    current_pos = 0
    for col in data_df.columns:
        policy_name = col.split('.')[0].replace('_table', '')
        if policy_name not in policy_positions:
            policy_positions[policy_name] = current_pos + (data_df.columns.str.startswith(policy_name + '.').sum() - 1) / 2
        current_pos += 1

    # Replace None in description
    safe_descriptions = np.where(pd.isna(descriptions_df.values), "", descriptions_df.values)

    def insert_line_breaks(text, max_line_length=60):
        if not isinstance(text, str):
            return ""
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "<br>".join(lines)

    # Process model card content if provided
    section_contents = {}
    if model_card_content:
        lines = model_card_content.strip().split('\n')
        for line in lines[2:]:
            if line.startswith('|'):
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) >= 2:
                    section_name = cells[0]
                    content = cells[1]
                    section_contents[section_name] = content

    # Apply line wrapping to descriptions
    wrapped_descriptions = np.vectorize(insert_line_breaks)(safe_descriptions)

    # === Y-axis label wrapping ===
    def wrap_label(label, max_len=20):
        words = label.split()
        lines = []
        current = ''
        for word in words:
            if len(current) + len(word) + 1 <= max_len:
                if current:
                    current += ' ' + word
                else:
                    current = word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return '<br>'.join(lines)
    wrapped_y_labels = [wrap_label(lbl) for lbl in data_df.index]

    # === X-axis: Only show policy names, centered ===
    # Find the center index for each policy's articles
    policy_to_indices = {}
    for idx, col in enumerate(data_df.columns):
        policy = col.split('.')[0].replace('_table', '')
        if policy not in policy_to_indices:
            policy_to_indices[policy] = []
        policy_to_indices[policy].append(idx)
    x_tickvals = []
    x_ticktext = []
    for policy, indices in policy_to_indices.items():
        center = int(np.mean(indices))
        x_tickvals.append(center)
        x_ticktext.append(policy)

    heatmap = go.Heatmap(
        z=data_df.values,
        x=list(range(len(data_df.columns))),  # Use numeric indices for x-axis
        y=wrapped_y_labels,  # Use wrapped y labels
        zmin=0,
        zmax=5,
        xgap=3, 
        ygap=3,
        colorscale = [[0.0, 'rgb(255,255,204)'], 
                    [0.2, 'rgb(255,255,204)'],
                    [0.2, 'rgb(161,218,180)'],
                    [0.4, 'rgb(161,218,180)'],
                    [0.4, 'rgb(100,181,205)'],
                    [0.6, 'rgb(100,181,205)'],
                    [0.6, 'rgb(54,130,189)'],
                    [0.8, 'rgb(54,130,189)'],
                    [0.8, 'rgb(8,88,158)'],
                    [1.0, 'rgb(8,88,158)']],
        hoverinfo="skip",  # <-- IMPORTANT: disable hover on heatmap so hover won't be covered
        showscale=False
    )

    # === Scatter overlay for hover ===
    hover_x = []
    hover_y = []
    hover_text = []

    # Create customdata array
    customdata = []
    for i, row in enumerate(data_df.index):
        for j, col in enumerate(data_df.columns):
            description = wrapped_descriptions[i, j]
            section_content = section_contents.get(row, "No content available")
            wrapped_section_content = insert_line_breaks(section_content)
            if not description:
                description = "This section fully complies with this article, so no reasoning is added"
            # Add full section name to hover text
            hover_x.append(j)
            hover_y.append(wrapped_y_labels[i])
            hover_str = f"<b>Section:</b> {row}<br><b>Article:</b> {col}<br><b>Score:</b> {data_df.values[i,j]}<br><b>Reasoning:</b> {description}<br><b>Section Content:</b> {wrapped_section_content}"
            hover_text.append(hover_str)

    scatter_hover = go.Scatter(
        x=hover_x,
        y=hover_y,
        mode='markers',
        marker=dict(size=20, opacity=0),  # Invisible markers
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    )

    # === Combine traces ===
    fig = go.Figure(data=[heatmap, scatter_hover])

    # === Add vertical lines for policy boundaries using add_vline ===
    policy_names = list(policy_to_indices.keys())
    for i in range(1, len(policy_names)):
        prev_indices = policy_to_indices[policy_names[i-1]]
        boundary = max(prev_indices) + 0.5
        fig.add_vline(
            x=boundary,
            line_width=2,
            line_dash="dash",
            line_color="rgba(0,0,0,0.3)",
            layer="above"
        )

    # === Layout ===
    fig.update_layout(
        title=None,
        xaxis=dict(
            showticklabels=True,
            showline=True,
            zeroline=False,
            side='top',
            tickvals=x_tickvals,  # Only show ticks at policy centers
            ticktext=x_ticktext,  # Only show policy names    

            # title='Policy',  # Removed axis title
        ),
        yaxis=dict(
            showticklabels=True,
            showline=True,
            zeroline=False,
            autorange='reversed',
            tickfont = dict(size=10),
        ),
        autosize = True,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        hoverlabel=dict(
            font_size=10,
            bgcolor="rgba(50, 50, 50, 0.95)",
            bordercolor="white"
        )
    )


    # Save to HTML file in static folder
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, output_filename)
    
    # Save with custom JavaScript
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        full_html=True,
        include_mathjax='cdn'
    )
    html_content = html_content.replace("<head>", "<head><style>html, body {margin: 0; padding: 0;}</style>")

    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"✅ Interactive heatmap saved to {output_path}")
    return output_filename

async def generate_policy_summary(policy_name, policy_data, model_card_content):
    """Generate a summary of policy compliance evaluation results"""
    # Read the prompt template
    async with aiofiles.open("prompt_summarize.txt", "r") as f:
        prompt_template = await f.read()

    # Format the evaluation results for the prompt
    evaluation_results = []
    for section in sections:
        for article, score in policy_data['scores'][section].items():
            if score < 5:  # Only include non-compliant items
                description = policy_data['descriptions'][section].get(article, "No description available")
                evaluation_results.append({
                    "section": section,
                    "article": article,
                    "score": score,
                    "description": description
                })

    # Format the evaluation results as a string
    evaluation_str = json.dumps(evaluation_results, indent=2)

    # Prepare the prompt
    prompt = prompt_template.replace("{{POLICY_DOC_NAME}}", policy_name)
    prompt = prompt.replace("{{EVALUATION_RESULT}}", evaluation_str)

    # Get summary from Claude
    response = llm.invoke(prompt)
    return response.content.strip()

async def generate_top_level_summary(policy_summaries):
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

async def generate_section_summary(section_name, section_data):
    """Generate a summary of compliance evaluation results for a specific model card section"""
    if TESTING_MODE:
        # Sample responses for testing mode
        sample_responses = {
                "System Name": """{
            "Overall": "#### Issues and Fixes:\\n- **Ambiguous system name**  \\n↳ *Clarify the system name to avoid confusion with other tools or versions.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Missing unique identifier**  \\n↳ *Include a distinct and traceable system name or code to support regulatory filing.*",
            "AIDA": "#### Issues and Fixes:\\n- **Inconsistent system labeling**  \\n↳ *Ensure the system name is uniform across all public and internal documentation.*",
            "CCPA": "#### Issues and Fixes:\\n- **Name not tied to consumer-facing functionality**  \\n↳ *Clearly indicate which services use this AI system so consumers understand its presence.*"
            }""",

                "Versioning Information": """{
            "Overall": "#### Issues and Fixes:\\n- **No version history provided**  \\n↳ *Add a versioning scheme and log of major changes to the system.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Insufficient version traceability**  \\n↳ *Document each deployed version to support accountability during audits.*",
            "AIDA": "#### Issues and Fixes:\\n- **No link between updates and risk**  \\n↳ *Explain how version updates are assessed for potential risk impacts.*",
            "CCPA": "#### Issues and Fixes:\\n- **Consumer-impacting updates unclear**  \\n↳ *Highlight changes that affect data use, privacy, or user-facing behavior.*"
            }""",

                "Primary Developer/Org": """{
            "Overall": "#### Issues and Fixes:\\n- **Missing organization accountability**  \\n↳ *Specify the developing organization and responsible departments.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Provider identity vague**  \\n↳ *Clearly state the system provider and outline their responsibilities.*",
            "AIDA": "#### Issues and Fixes:\\n- **Lack of developer governance info**  \\n↳ *Include governance structures and responsible individuals for oversight.*",
            "CCPA": "#### Issues and Fixes:\\n- **Contact identity missing**  \\n↳ *Ensure users know who operates the system and how to reach them.*"
            }""",

                "Contact Info": """{
            "Overall": "#### Issues and Fixes:\\n- **Missing support channel**  \\n↳ *Provide a clear contact for technical support and ethical concerns.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No incident reporting mechanism**  \\n↳ *Add a process for users to report potential harm or system failures.*",
            "AIDA": "#### Issues and Fixes:\\n- **Accountability contact undefined**  \\n↳ *Specify a person or team responsible for compliance communications.*",
            "CCPA": "#### Issues and Fixes:\\n- **No contact for data requests**  \\n↳ *Include a channel for users to request data access or deletion.*"
            }""",
                "System Overview": """{
            "Overall": "#### Issues and Fixes:\\n- **High-level functionality unclear**  \\n↳ *Include a concise summary of what the AI system does and its boundaries.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Missing risk categorization**  \\n↳ *State whether the system qualifies as high-risk under Annex III of the EU AI Act.*",
            "AIDA": "#### Issues and Fixes:\\n- **No reference to intended impact**  \\n↳ *Describe the anticipated effects on individuals and society as required by AIDA.*",
            "CCPA": "#### Issues and Fixes:\\n- **Lacks mention of user data flow**  \\n↳ *Clarify how personal data moves through the system, if applicable.*"
            }""",

                "Primary intended uses": """{
            "Overall": "#### Issues and Fixes:\\n- **Intended use too vague**  \\n↳ *Clarify the real-world tasks or decisions the system is meant to support.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No mapping to risk categories**  \\n↳ *Indicate whether any intended uses relate to high-risk applications under the regulation.*",
            "AIDA": "#### Issues and Fixes:\\n- **Societal impact not articulated**  \\n↳ *Describe how these use cases may influence people’s rights or access to services.*",
            "CCPA": "#### Issues and Fixes:\\n- **Use cases lack privacy dimension**  \\n↳ *Explain how each use case relates to data collection or user profiling.*"
            }""",

                "Primary intended users": """{
            "Overall": "#### Issues and Fixes:\\n- **User roles not defined**  \\n↳ *Specify who the system is designed for—experts, consumers, or institutions.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No human oversight mapping**  \\n↳ *Explain how user roles affect control over the AI system.*",
            "AIDA": "#### Issues and Fixes:\\n- **Accessibility needs not considered**  \\n↳ *Account for inclusion of marginalized or underserved user groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **End-user data rights undefined**  \\n↳ *Clarify how user roles affect data visibility or deletion options.*"
            }""",

                "Out-of-scope use cases": """{
            "Overall": "#### Issues and Fixes:\\n- **No constraints listed**  \\n↳ *Document scenarios where system use is not advised or disallowed.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No safeguards against misuse**  \\n↳ *Describe restrictions that prevent the AI system from being used in prohibited contexts.*",
            "AIDA": "#### Issues and Fixes:\\n- **Potential for repurposing unacknowledged**  \\n↳ *List common misuses and provide warnings or disclaimers.*",
            "CCPA": "#### Issues and Fixes:\\n- **No restriction on behavioral tracking**  \\n↳ *Clarify that system is not intended for unconsented behavioral analytics.*"
            }""",

                "Terms and conditions": """{
            "Overall": "#### Issues and Fixes:\\n- **Missing licensing terms**  \\n↳ *Add legal conditions under which the system can be accessed and used.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **User responsibilities undefined**  \\n↳ *Define who is responsible for compliance depending on deployment context.*",
            "AIDA": "#### Issues and Fixes:\\n- **No liability disclaimer**  \\n↳ *Include legal notices about AI-related harm and mitigation duties.*",
            "CCPA": "#### Issues and Fixes:\\n- **No consent-related language**  \\n↳ *State how consent is obtained or revoked under usage terms.*"
            }""",

                "Current legal compliance status": """{
            "Overall": "#### Issues and Fixes:\\n- **No mention of applicable regulations**  \\n↳ *List which data, consumer, or AI regulations the system adheres to.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No reference to conformity assessment**  \\n↳ *Document status of required assessments under EU AI Act Title III.*",
            "AIDA": "#### Issues and Fixes:\\n- **Lack of risk-based compliance review**  \\n↳ *Note if the system underwent algorithmic impact assessment (AIA).*",
            "CCPA": "#### Issues and Fixes:\\n- **No disclosure of privacy policies**  \\n↳ *State whether the system has been reviewed for CCPA compliance.*"
            }""",

                "Dataset Description": """{
            "Overall": "#### Issues and Fixes:\\n- **Data origin unclear**  \\n↳ *Provide source details and licensing terms of the datasets used.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No discussion of representativeness**  \\n↳ *Explain how datasets reflect the population where the system is used.*",
            "AIDA": "#### Issues and Fixes:\\n- **Unclear data collection process**  \\n↳ *Indicate whether data was collected directly, indirectly, or scraped.*",
            "CCPA": "#### Issues and Fixes:\\n- **Lack of personal data flagging**  \\n↳ *Identify if any data used is considered personal under CCPA definitions.*"
            }""",

                "Collection Method": """{
            "Overall": "#### Issues and Fixes:\\n- **Unspecified collection procedure**  \\n↳ *Describe how and where data was collected, including devices or APIs used.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Collection timeframe omitted**  \\n↳ *Specify when the data was obtained, as timing affects validity.*",
            "AIDA": "#### Issues and Fixes:\\n- **No consent method disclosed**  \\n↳ *Explain how user consent was obtained for data collection.*",
            "CCPA": "#### Issues and Fixes:\\n- **No opt-out capability for users**  \\n↳ *Document if users could reject participation or data tracking.*"
            }""",

                "Bias Mitigation Measures": """{
            "Overall": "#### Issues and Fixes:\\n- **Limited bias reduction explanation**  \\n↳ *Provide details on pre-processing, in-processing, or post-processing methods.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No fairness audits described**  \\n↳ *Include results from any bias testing required for high-risk systems.*",
            "AIDA": "#### Issues and Fixes:\\n- **No documentation of systemic risk**  \\n↳ *Explain whether the model was reviewed for disproportionate harm.*",
            "CCPA": "#### Issues and Fixes:\\n- **Protected group flags missing**  \\n↳ *Indicate if system treats sensitive data differently to mitigate bias.*"
            }""",

                "Usage Constraints": """{
            "Overall": "#### Issues and Fixes:\\n- **System limits not defined**  \\n↳ *Clarify technical and policy constraints on system usage.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No usage boundary conditions**  \\n↳ *Define operating conditions to prevent deployment beyond scope.*",
            "AIDA": "#### Issues and Fixes:\\n- **Lack of policy-driven usage checks**  \\n↳ *Specify measures that prevent harmful overuse or repurposing.*",
            "CCPA": "#### Issues and Fixes:\\n- **No data use limitations documented**  \\n↳ *Clearly list what data may not be used or retained under CCPA.*"
            }""",

                "Summary of Performance Assessment": """{
            "Overall": "#### Issues and Fixes:\\n- **No baseline or benchmark provided**  \\n↳ *Add quantitative metrics with comparisons to industry baselines.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Performance for critical tasks unclear**  \\n↳ *Report accuracy, robustness, and reliability for key functions.*",
            "AIDA": "#### Issues and Fixes:\\n- **Impact of errors unaddressed**  \\n↳ *Discuss how model performance affects individuals or groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **No indication of error in personal data usage**  \\n↳ *Include performance metrics relevant to privacy and personalization.*"
            }""",

                "Disaggregated Performance": """{
            "Overall": "#### Issues and Fixes:\\n- **Subgroup performance not reported**  \\n↳ *Present how the system performs across different demographic or user groups.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No fairness performance shown**  \\n↳ *Include subgroup error rates as required for high-risk classification.*",
            "AIDA": "#### Issues and Fixes:\\n- **Equity implications not explored**  \\n↳ *Report whether performance gaps might cause harm or exclusion.*",
            "CCPA": "#### Issues and Fixes:\\n- **Demographic bias impact not measured**  \\n↳ *Evaluate if personalization varies by age, race, or other protected attributes.*"
            }""",

                "Testing Contexts": """{
            "Overall": "#### Issues and Fixes:\\n- **Unclear test environment**  \\n↳ *List environments and inputs used during testing, including edge cases.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Real-world conditions missing**  \\n↳ *Simulate and document tests under intended operational settings.*",
            "AIDA": "#### Issues and Fixes:\\n- **Testing fails to simulate harms**  \\n↳ *Use representative conditions that reflect potential risk.*",
            "CCPA": "#### Issues and Fixes:\\n- **Data flow not validated in tests**  \\n↳ *Ensure tests cover scenarios involving personal data handling.*"
            }""",

                "Evaluations for Edge Cases or Adversarial Inputs": """{
            "Overall": "#### Issues and Fixes:\\n- **Limited adversarial testing**  \\n↳ *Expand test cases to include abnormal, unexpected, or hostile inputs.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Robustness testing incomplete**  \\n↳ *Evaluate resilience to edge cases as part of high-risk system obligations.*",
            "AIDA": "#### Issues and Fixes:\\n- **Potential for harm under-tested**  \\n↳ *Identify how unusual inputs may cause discriminatory or dangerous outcomes.*",
            "CCPA": "#### Issues and Fixes:\\n- **Adversarial misuse affecting personal data unaddressed**  \\n↳ *Include tests for manipulative attacks that affect privacy or output.*"
            }""",

                "Potential Risks and Harms": """{
            "Overall": "#### Issues and Fixes:\\n- **No risk assessment framework provided**  \\n↳ *List foreseeable risks, likelihoods, and severity, with mitigation strategies.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Annex III risk types not mapped**  \\n↳ *Align identified risks with EU AI Act high-risk categories.*",
            "AIDA": "#### Issues and Fixes:\\n- **Individual rights impact vague**  \\n↳ *Explain how the system may affect autonomy, dignity, or social inclusion.*",
            "CCPA": "#### Issues and Fixes:\\n- **No disclosure of potential data misuse**  \\n↳ *Describe possible misuse of personal data and safeguards in place.*"
            }""",

                "Actions taken": """{
            "Overall": "#### Issues and Fixes:\\n- **Mitigations not linked to identified risks**  \\n↳ *Clearly show how specific actions address known risks or deficiencies.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Corrective actions not documented**  \\n↳ *Include risk reduction efforts taken during design and testing phases.*",
            "AIDA": "#### Issues and Fixes:\\n- **Unclear alignment with accountability obligations**  \\n↳ *Highlight ongoing governance or improvement steps tied to compliance.*",
            "CCPA": "#### Issues and Fixes:\\n- **No action noted on user complaints**  \\n↳ *Mention any steps taken in response to access, correction, or deletion requests.*"
            }""",

                "Misuse Scenarios": """{
            "Overall": "#### Issues and Fixes:\\n- **Misuse potential not documented**  \\n↳ *Describe how the system could be exploited or misunderstood.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No prohibited use warnings**  \\n↳ *Include disclaimers for banned or high-risk uses, such as surveillance.*",
            "AIDA": "#### Issues and Fixes:\\n- **Misuse detection strategies missing**  \\n↳ *Explain how the system monitors or responds to unintended applications.*",
            "CCPA": "#### Issues and Fixes:\\n- **Data misuse not discussed**  \\n↳ *Identify how unauthorized access or repurposing of personal data is prevented.*"
            }""",

                "Human Oversight": """{
            "Overall": "#### Issues and Fixes:\\n- **Oversight mechanisms poorly defined**  \\n↳ *Specify when and how humans can intervene in system operation.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No oversight protocol for high-risk decisions**  \\n↳ *Establish human review and override paths for sensitive outcomes.*",
            "AIDA": "#### Issues and Fixes:\\n- **Unclear if humans have meaningful control**  \\n↳ *Document how human judgment supplements or monitors automation.*",
            "CCPA": "#### Issues and Fixes:\\n- **Lack of manual appeal process for users**  \\n↳ *Ensure users can request human handling of automated decisions.*"
            }""",

                "Update Frequency": """{
            "Overall": "#### Issues and Fixes:\\n- **Update cadence not disclosed**  \\n↳ *Describe how often and under what conditions the model or system is updated.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No monitoring triggers for retraining**  \\n↳ *Indicate how performance drift or data changes lead to updates.*",
            "AIDA": "#### Issues and Fixes:\\n- **No link between updates and risk**  \\n↳ *Explain how update decisions incorporate harm prevention logic.*",
            "CCPA": "#### Issues and Fixes:\\n- **Versioning not tied to data retention**  \\n↳ *Clarify whether updates reset or affect data collection timelines.*"
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

async def run_ai_pipeline(model_card_path, policy_folder, output_path, selected_policies=None):
    try:
        # Load model card content
        async with aiofiles.open(model_card_path, "r", encoding="utf-8") as f:
            model_card_content = await f.read()

        # Read prompt template
        prompt_template_path = "prompt.txt"
        async with aiofiles.open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = await f.read()

        # Get list of policy files and filter based on selection
        policy_files = sorted(os.listdir(policy_folder))
        if selected_policies:
            policy_files = [f for f in policy_files if f.split('.')[0] in selected_policies]
            if not policy_files:
                raise ValueError("No valid policies selected")
        
        report_lines = []
        
        # Initialize dictionary to store all data
        all_policy_data = {}
        
        # Initialize dictionary to store section-based data
        section_data = {section: {} for section in sections}
        
        for policy_file in policy_files:
            try:
                policy_path = os.path.join(policy_folder, policy_file)
                async with aiofiles.open(policy_path, "r") as pf:
                    legal_doc_content = await pf.read()
                if not TESTING_MODE:
                    # Get chunking strategy for this policy
                    chunking_prompt = get_chunking_prompt().replace("{{POLICY_DOC}}", legal_doc_content)
                    chunk_response = llm.invoke(chunking_prompt).content
                    print("\nChunking response:", chunk_response)
                    chunks = parse_chunk_response(chunk_response)
                    print(f"Policy {policy_file} will be evaluated in {len(chunks)} chunks")
                    print("Chunks:", chunks)

                # Initialize section data for this policy
                policy_section_scores = {section: {} for section in sections}
                policy_section_descriptions = {section: {} for section in sections}

                for section in sections:
                    if TESTING_MODE:
                        # Set the context for FakeLLM
                        fakeLlm.set_context(policy_file.split('.')[0], section)
                        
                        # Create the prompt without chunking
                        chunk_prompt = (
                            prompt_template
                            .replace("{{MODEL_CARD}}", model_card_content)
                            .replace("{{LEGAL_DOC}}", legal_doc_content)
                            .replace("{{SECTION}}", section)
                            .replace("{{START_ART}}", "1")  # Use dummy values since we're not chunking
                            .replace("{{END_ART}}", "999")  # Use dummy values since we're not chunking
                        )
                        
                        print(f"Evaluating {policy_file} section '{section}' in testing mode...")
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
                        response = fakeLlm.invoke(messages)
                        print(f'fakeLlm response is: {response.content}')

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
                    else:
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

        # Create combined DataFrames
        # First, create a list of all columns (policy.article combinations)
        all_columns = []
        for policy_name, policy_data in all_policy_data.items():
            print(f"\nDebug - Processing columns for {policy_name}")
            print(f"Available articles:", policy_data['articles'])
            for article in policy_data['articles']:
                # Ensure consistent article naming format
                column = f"{policy_name}.Art.{article}"
                print(f"Debug - Adding column: {column}")
                all_columns.append(column)

        print("\nDebug - Final column list:", all_columns)
        
        # Create empty DataFrames
        scores_df = pd.DataFrame(index=sections, columns=all_columns)
        descriptions_df = pd.DataFrame(index=sections, columns=all_columns)

        # Fill in the data
        for policy_name, policy_data in all_policy_data.items():
            for section in sections:
                for article in policy_data['articles']:
                    column = f"{policy_name}.Art.{article}"
                    # Remove any 'Art.' prefix from the article number when accessing the data
                    article_key = article.replace('Art.', '').strip()
                    
                    # Debug print for problematic articles
                    if article in ['6', '2'] and policy_name in ['AIDA_table', 'EU_table']:
                        print(f"\nDebug - Processing problematic article:")
                        print(f"Policy: {policy_name}")
                        print(f"Article: {article}")
                        print(f"Column name: {column}")
                        print(f"Article key: {article_key}")
                        print(f"Available scores: {policy_data['scores'][section].keys()}")
                    
                    scores_df.loc[section, column] = policy_data['scores'][section].get(article_key, 0)
                    descriptions_df.loc[section, column] = policy_data['descriptions'][section].get(article_key, "No evaluation")

        # Print parsed data information
        print("\n=== Parsed Scores Data ===")
        print("\nScores DataFrame Structure:")
        print(scores_df.info())
        print("\nScores DataFrame Contents:")
        print(scores_df)
        print("\nScores Shape:", scores_df.shape)
        print("=== End of Scores Data ===\n")

        print("\n=== Parsed Descriptions Data ===")
        print("\nDescriptions DataFrame Structure:")
        print(descriptions_df.info())
        print("\nDescriptions DataFrame Contents:")
        print(descriptions_df)
        print("\nDescriptions Shape:", descriptions_df.shape)
        print("=== End of Descriptions Data ===\n")
        
        # Generate heatmaps for each section
        heatmap_filenames = []
        # Create a single combined heatmap instead of individual ones
        timestamp = int(time.time())
        heatmap_filename = f"heatmap_combined_{timestamp}.html"
        # Reindex DataFrames to match the order of the 'sections' list
        scores_df = scores_df.reindex(sections)
        descriptions_df = descriptions_df.reindex(sections)
        generate_interactive_heatmap(scores_df, descriptions_df, "Combined", model_card_content, heatmap_filename)
        heatmap_filenames.append(heatmap_filename)

        # Generate summaries for each policy
        summaries = {}
        for policy_file in policy_files:
            try:
                policy_name = policy_file.split('.')[0]
                if policy_name in all_policy_data:
                    policy_data = all_policy_data[policy_name]
                    print(f"\nGenerating summary for policy: {policy_name}")
                    summary = await generate_policy_summary(policy_name, policy_data, model_card_content)
                    summaries[policy_name] = summary
                    print(f"Generated summary for {policy_name}")
            except Exception as e:
                print(f"Error generating summary for policy {policy_name}: {str(e)}")
                continue

        print("\nGenerated summaries for policies:", list(summaries.keys()))
        
        # Generate top-level summary
        print("\nGenerating top-level summary across all policies...")
        top_level_summary = await generate_top_level_summary(summaries)
        print("Generated top-level summary")

        # Generate section-based summaries
        section_summaries = {}
        for section in sections:
            try:
                if not section_data[section]:  # Check if there's no data for this section
                    section_summaries[section] = f"""#### ⚠️ {section} – No Evaluation Data

Note: No evaluation data was provided for this section. This could indicate that:
- The section is missing from the model card
- No applicable policy requirements were found
- An error occurred during evaluation

Please ensure this section exists and contains the necessary information."""
                else:
                    summary = await generate_section_summary(section, section_data[section])
                    section_summaries[section] = summary
                    print(f"Generated summary for section: {section}")
            except Exception as e:
                print(f"Error generating summary for section {section}: {str(e)}")
                section_summaries[section] = f"""#### ❌ {section} – Error

An error occurred while generating the summary for this section. Please check the logs for more details."""

        print("All evaluations completed.")
        return heatmap_filenames, summaries, top_level_summary, section_summaries
    except Exception as e:
        print(f"Error in AI pipeline: {str(e)}")
        raise