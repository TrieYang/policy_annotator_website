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

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize Claude model
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    #model="claude-3-haiku-20240307",
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0.3
)

# Section names for iteration
sections = [
    "System Name",
    "Versioning Information",
    "Primary Developer/Org",
    "Contact Info",
    "System Overview",
    "Primary intended uses",
    "Primary intended users",
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

    # Create customdata array
    customdata = []
    for i, row in enumerate(data_df.index):
        row_data = []
        for j, col in enumerate(data_df.columns):
            description = wrapped_descriptions[i, j]
            section_content = section_contents.get(row, "No content available")
            wrapped_section_content = insert_line_breaks(section_content)
            # Include full article reference in hover data
            row_data.append([description, wrapped_section_content, col])
        customdata.append(row_data)
    customdata = np.array(customdata)

    fig = go.Figure(data=go.Heatmap(
        z=data_df.values,
        x=list(range(len(data_df.columns))),  # Use numeric indices for x-axis
        y=data_df.index,
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
        hoverongaps=False,
        customdata=customdata,
        hovertemplate=(
            "Section: %{y}<br>" +
            "Article: %{customdata[2]}<br>" +  # Show full article reference in hover
            "Score: %{z}<br>" +
            "Description: %{customdata[0]}" +
            "<br><br><b>Model Card Content:</b><br>%{customdata[1]}<extra></extra>"
        ),
        showscale=False
    ))

    # Update layout with custom axis labels
    fig.update_layout(
        title=None,
        xaxis=dict(
            showticklabels=False,  # Hide x-axis labels
            showline=False,  # Hide x-axis line
            zeroline=False,  # Hide zero line
            side='bottom'
        ),
        yaxis=dict(
            showticklabels=False,  # Hide y-axis labels since they're redundant
            showline=False,  # Hide y-axis line
            zeroline=False   # Hide zero line
        ),
        width=900,
        height=30,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.show(config={'displayModeBar': False})

    # Add more visual enhancements
    fig.update_traces(
        dict(
            showscale=False  # Remove colorbar
        )
    )

    # Save to HTML file in static folder
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, output_filename)
    
    # Save with custom JavaScript
    html_content = fig.to_html(include_plotlyjs='cdn', full_html=True, include_mathjax='cdn')
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
    # Read the prompt template
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
                        print(response.content)
                        print(response.usage_metadata["input_token_details"])

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
                print(f"Debug - Sorted articles for {policy_name}:", sorted_articles)
                
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
        for section in sections:
            # Create DataFrames for just this section
            section_scores_df = pd.DataFrame(index=[section], columns=all_columns)
            section_descriptions_df = pd.DataFrame(index=[section], columns=all_columns)
            
            # Fill in the data for this section
            for policy_name, policy_data in all_policy_data.items():
                for article in policy_data['articles']:
                    column = f"{policy_name}.Art.{article}"
                    article_key = article.replace('Art.', '').strip()
                    section_scores_df.loc[section, column] = policy_data['scores'][section].get(article_key, 0)
                    section_descriptions_df.loc[section, column] = policy_data['descriptions'][section].get(article_key, "No evaluation")
            
            # Generate heatmap for this section
            timestamp = int(time.time())
            # Sanitize section name for filename by replacing problematic characters
            safe_section_name = section.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
            heatmap_filename = f"heatmap_{safe_section_name}_{timestamp}.html"
            generate_interactive_heatmap(section_scores_df, section_descriptions_df, section, model_card_content, heatmap_filename)
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