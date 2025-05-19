import os
import time
import json
from dotenv import load_dotenv
import aiofiles
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        colorscale=[
            [0.0, '#313695'],   # Score 0 – deep blue (fully compliant)
            [0.2, '#74add1'],   # Score 1 – blue
            [0.4, '#abd9e9'],   # Score 2 – light blue
            [0.6, '#fdae61'],   # Score 3 – orange
            [0.8, '#d73027'],   # Score 4 – red-orange
            [1.0, '#a50026'],   # Score 5 – deep red (violation)
        ],
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
            title='Policy',
            showticklabels=True,
            showline=True,
            zeroline=False,
            side='top',
            tickvals=x_tickvals,  # Only show ticks at policy centers
            ticktext=x_ticktext,  # Only show policy names    

            # title='Policy',  # Removed axis title
        ),
        yaxis=dict(
            title='Model Card Section',
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
