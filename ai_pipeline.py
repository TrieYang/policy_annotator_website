import os
import time
import sys
import json
import importlib.util
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import aiofiles
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from policy_chunker import get_chunking_prompt, parse_chunk_response, get_section_groups
from utils.section_summary import generate_section_summary, generate_multiple_section_summaries
from utils.top_level_summary import generate_top_level_summary
from utils.interactive_heatmap import generate_interactive_heatmap
from utils.policy_summary import generate_policy_summary
import asyncio

SAMPLE_RESPONSE_FOLDER = "./sample_responses"
TESTING_MODE = False

def load_relevancy_map(policy_name):
    """
    Dynamically load the appropriate relevancy map based on policy name.
    Returns the section_chunks dictionary from the corresponding relevancy map file.
    """
    try:
        # Convert policy name to lowercase for file matching
        policy_lower = policy_name.lower()
        
        # Map policy names to relevancy map files
        if 'eu' in policy_lower:
            map_file = "relevancy_maps/eu_relevancy_map.py"
        elif 'ccpa' in policy_lower:
            map_file = "relevancy_maps/ccpa_relevancy_map.py"
        elif 'aida' in policy_lower:
            map_file = "relevancy_maps/aida_relevancy_map.py"
        elif 'colorado' in policy_lower:
            map_file = "relevancy_maps/colorado_relevancy_map.py"
        elif 'gdpr' in policy_lower:
            map_file = "relevancy_maps/gdpr_relevancy_map.py"
        else:
            # Default fallback - use the current hardcoded map
            print(f"Warning: No specific relevancy map found for policy '{policy_name}', using default map")
            return {
                'System Name': [('Article 2', 'Article 4', 'Article 5', 'Article 7', 'Article 8')],
                'Versioning Information': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Primary Developer/Org': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Contact Info': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'System Overview': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Primary intended uses': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Primary intended users': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Out-of-scope use cases': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Terms and conditions': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Current legal compliance status': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Dataset Description': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Collection Method': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Bias Mitigation Measures': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Usage Constraints': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Summary of Performance Assessment': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Disaggregated Performance': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Testing Contexts': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Evaluations for Edge Cases or Adversarial Inputs': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Potential Risks and Harms': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Actions taken': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Misuse Scenarios': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Human Oversight': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
                'Update Frequency': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')]
            }
        
        # Load the relevancy map module
        spec = importlib.util.spec_from_file_location("relevancy_map", map_file)
        relevancy_map_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relevancy_map_module)
        
        print(f"Loaded relevancy map for policy: {policy_name}")
        return relevancy_map_module.section_chunks
        
    except Exception as e:
        print(f"Error loading relevancy map for policy '{policy_name}': {e}")
        # Return the default hardcoded map as fallback
        return {
            'System Name': [('Article 2', 'Article 4', 'Article 5', 'Article 7', 'Article 8')],
            'Versioning Information': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Primary Developer/Org': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Contact Info': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'System Overview': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Primary intended uses': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Primary intended users': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Out-of-scope use cases': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Terms and conditions': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Current legal compliance status': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Dataset Description': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Collection Method': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Bias Mitigation Measures': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Usage Constraints': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Summary of Performance Assessment': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Disaggregated Performance': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Testing Contexts': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Evaluations for Edge Cases or Adversarial Inputs': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Potential Risks and Harms': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Actions taken': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Misuse Scenarios': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Human Oversight': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')],
            'Update Frequency': [('Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5', 'Article 6', 'Article 7', 'Article 8', 'Article 9', 'Article 10'), ('Article 11', 'Article 12', 'Article 13', 'Article 14', 'Article 15', 'Article 16', 'Article 17', 'Article 18', 'Article 19', 'Article 20'), ('Article 21', 'Article 22', 'Article 23', 'Article 24', 'Article 25', 'Article 26', 'Article 27', 'Article 28', 'Article 29', 'Article 30'), ('Article 31', 'Article 32', 'Article 33', 'Article 34', 'Article 35', 'Article 36', 'Article 37', 'Article 38', 'Article 39', 'Article 40')]
        }

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
model="claude-sonnet-4-20250514",
#model="claude-3-7-sonnet-20250219",
#model="claude-3-haiku-20240307",
anthropic_api_key=ANTHROPIC_API_KEY,
temperature=0.3,
max_retries=3,  # Add retry logic
timeout=300)  # Add timeout

# Section names for iteration
sections = [
    "System Name",
    "Versioning Information",
    "Primary Developer/Org",
    "Contact Information",
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



async def run_ai_pipeline(model_card_path, policy_folder, output_path, selected_policies=None):
    try:
        # Load model card content from CSV and convert to markdown table format
        df = pd.read_csv(model_card_path)
        
        # Create markdown table content
        markdown_content = "| Section | Content |\n| ------- | ------- |\n"
        
        # Iterate through rows and create markdown table rows
        for _, row in df.iterrows():
            section = row['Section']
            content = row['Your Response']
            if pd.notna(section) and pd.notna(content):  # Skip empty rows
                markdown_content += f"| {section} | {content} |\n"
        
        model_card_content = markdown_content
        print(model_card_content)

        # Read prompt template
        prompt_template_path = "new_prompt_second.txt"
        async with aiofiles.open(prompt_template_path, "r", encoding="utf-8") as f:
            chunk_prompt = await f.read()

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
        
        # Create a list to store all concurrent tasks
        all_tasks = []
        task_metadata = {}  # To track which task corresponds to which policy/section/chunk
        
        print(f"\n{'='*50}")
        print(f"Preparing {len(policy_files)} policies for concurrent processing...")
        print(f"{'='*50}")
        
        # Prepare all tasks first without executing them
        for policy_file in policy_files:
            try:
                print(f"Preparing policy file: {policy_file}")
                
                policy_path = os.path.join(policy_folder, policy_file)
                
                # Check if file exists and is readable
                if not os.path.exists(policy_path):
                    print(f"Error: Policy file {policy_path} does not exist")
                    continue
                    
                file_size = os.path.getsize(policy_path)
                print(f"Policy file size: {file_size} bytes")
                
                async with aiofiles.open(policy_path, "r") as pf:
                    legal_doc_content = await pf.read()
                
                print(f"Successfully read policy file. Content length: {len(legal_doc_content)} characters")
      
                # Load the relevancy map for the current policy
                section_chunks = load_relevancy_map(policy_file.split('.')[0])
                print(f"Loaded relevancy map with {len(section_chunks)} sections")

                # Create tasks for all sections and chunks
                for section in sections:
                    # Get the chunks for this specific section
                    section_specific_chunks = section_chunks.get(section, [])
                    if not section_specific_chunks:
                        print(f"No chunks for section '{section}' - skipping evaluation")
                        continue
                        
                    print(f"Preparing {len(section_specific_chunks)} chunks for section '{section}'")
                    
                    for chunk_idx, chunk in enumerate(section_specific_chunks):
                        # Convert the chunk of articles into a comma-separated string
                        articles_str = ", ".join(chunk)
                        
                        # Replace placeholders in the prompt
                        formatted_prompt = (
                            chunk_prompt
                            .replace("{{LEGAL_DOC}}", legal_doc_content)
                            .replace("{{SECTION}}", section)
                            .replace("{{ARTICLES_TO_EVALUATE}}", articles_str)
                        )
                        
                        messages = [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{legal_doc_content}",
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                    {
                                        "type": "text",
                                        "text": f"{model_card_content}",
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                ],
                            },
                            {
                                "role": "user",
                                "content": f"{formatted_prompt}",
                            },
                        ]
                        
                        # Create the task
                        task = asyncio.create_task(
                            asyncio.wait_for(
                                asyncio.to_thread(llm.invoke, messages),
                                timeout=300  # 5 minute timeout for each evaluation
                            )
                        )
                        
                        # Store task metadata for later processing
                        task_id = len(all_tasks)
                        task_metadata[task_id] = {
                            'policy_file': policy_file,
                            'section': section,
                            'chunk': chunk,
                            'articles_str': articles_str,
                            'messages': messages
                        }
                        
                        all_tasks.append(task)
                        
            except Exception as e:
                print(f"Error preparing policy file {policy_file}: {str(e)}")
                continue
        
        print(f"\n{'='*50}")
        print(f"Executing {len(all_tasks)} LLM requests concurrently...")
        print(f"{'='*50}")
        
        # Execute all tasks concurrently
        start_time = time.time()
        responses = await asyncio.gather(*all_tasks, return_exceptions=True)
        end_time = time.time()
        
        print(f"Completed {len(all_tasks)} requests in {end_time - start_time:.2f} seconds")
        print(f"Average time per request: {(end_time - start_time) / len(all_tasks):.2f} seconds")
        
        # Initialize section data for all policies
        policy_section_scores = {}
        policy_section_descriptions = {}
        
        for policy_file in policy_files:
            policy_name = policy_file.split('.')[0]
            policy_section_scores[policy_name] = {section: {} for section in sections}
            policy_section_descriptions[policy_name] = {section: {} for section in sections}
        
        # Process all responses
        print(f"\n{'='*50}")
        print(f"Processing {len(responses)} responses...")
        print(f"{'='*50}")
        
        successful_requests = 0
        failed_requests = 0
        
        for task_id, response in enumerate(responses):
            metadata = task_metadata[task_id]
            policy_file = metadata['policy_file']
            section = metadata['section']
            chunk = metadata['chunk']
            articles_str = metadata['articles_str']
            
            if isinstance(response, Exception):
                print(f"Request failed for {policy_file} section '{section}' articles {articles_str}: {str(response)}")
                failed_requests += 1
                continue
            
            try:
                print(f"Processing response for {policy_file} section '{section}' articles {articles_str}")
                
                # Parse the markdown table to extract JSON content
                lines = response.content.strip().splitlines()

                # Helper function to detect markdown separator row
                def is_separator_row(line):
                    parts = [p.strip() for p in line.strip().split('|')[1:-1]]
                    return all(part.replace('-', '') == '' for part in parts)

                # Get all rows except the separator row (the one with |---|---| etc.)
                table_rows = [
                    line for line in lines
                    if line.strip().startswith('|') and not is_separator_row(line)
                ]
                
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
                        
                        policy_name = policy_file.split('.')[0]
                        policy_section_scores[policy_name][section][article_num] = json_data['score']
                        policy_section_descriptions[policy_name][section][article_num] = json_data.get('description', '')
                        
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"Error parsing row {data_row}: {e}")
                        continue
                
                successful_requests += 1
                
            except Exception as e:
                print(f"Error processing response for {policy_file} section '{section}' articles {articles_str}: {str(e)}")
                failed_requests += 1
                continue
        
        print(f"\n{'='*50}")
        print(f"Processing Summary:")
        print(f"Successful requests: {successful_requests}")
        print(f"Failed requests: {failed_requests}")
        print(f"Total requests: {len(all_tasks)}")
        print(f"{'='*50}")
        
        # Store the data for each policy
        for policy_file in policy_files:
            try:
                policy_name = policy_file.split('.')[0]
                
                # Get all article numbers and convert to float for proper sorting
                all_articles = set()
                for section in sections:
                    all_articles.update(policy_section_scores[policy_name][section].keys())
                
                print(f"Policy {policy_name}: Found {len(all_articles)} total articles across all sections")
                
                # Convert to float for sorting, handling both integer and decimal article numbers
                def article_to_sortable(art):
                    # Remove any 'Art.' prefix if it exists
                    clean_art = art.replace('Art.', '').strip()
                    try:
                        # First try converting to float for simple numbers
                        float_val = float(clean_art)
                        # Return as a single-element tuple for consistency
                        return (float_val,)
                    except ValueError:
                        # For complex article numbers like '1798.199.60', split by dots and convert each part
                        try:
                            parts = clean_art.split('.')
                            # Convert each part to float, handling any non-numeric parts
                            numeric_parts = []
                            for part in parts:
                                try:
                                    numeric_parts.append(float(part))
                                except ValueError:
                                    # If a part can't be converted to float, use 0
                                    numeric_parts.append(0.0)
                            # Return tuple for proper sorting of hierarchical numbers
                            return tuple(numeric_parts)
                        except Exception:
                            # If all else fails, return the original string as a single-element tuple
                            print(f"Warning: Could not convert article number '{clean_art}' to sortable format")
                            return (clean_art,)
                        
                # Sort articles using the custom sorting function
                sorted_articles = sorted(all_articles, key=article_to_sortable)                
                all_policy_data[policy_name] = {
                    'scores': policy_section_scores[policy_name],
                    'descriptions': policy_section_descriptions[policy_name],
                    'articles': sorted_articles
                }

                # After processing each policy, organize data by section
                for section in sections:
                    section_data[section][policy_name] = {
                        'scores': policy_section_scores[policy_name][section],
                        'descriptions': policy_section_descriptions[policy_name][section]
                    }
                
                print(f"Successfully processed policy {policy_name} with {len(sorted_articles)} articles")

            except Exception as e:
                print(f"Error processing policy file {policy_file}: {str(e)}")
                continue

        # Create combined DataFrames
        # First, create a list of all columns (policy.article combinations)
        all_columns = []
        for policy_name, policy_data in all_policy_data.items():
            print(f"Available articles:", policy_data['articles'])
            for article in policy_data['articles']:
                # Ensure consistent article naming format
                column = f"{policy_name}.Art.{article}"
                all_columns.append(column)
        
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
                    
                    scores_df.loc[section, column] = policy_data['scores'][section].get(article_key, 0)
                    descriptions_df.loc[section, column] = policy_data['descriptions'][section].get(article_key, "Irrelevant ")
        
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
                    print(f"Policy data keys: {list(policy_data.keys())}")
                    print(f"Number of sections with data: {len([s for s in sections if s in policy_data['scores']])}")
                    
                    # Validate policy data structure
                    if 'scores' not in policy_data or 'descriptions' not in policy_data:
                        print(f"Error: Invalid policy data structure for {policy_name}")
                        summaries[policy_name] = f"""**Current Compliance Status:**
Error: Invalid policy data structure for {policy_name}. Please check the evaluation results manually.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| Data Structure Error | Invalid policy data structure | Review evaluation process | High |"""
                        continue
                    
                    # Check if there's any actual evaluation data
                    has_data = False
                    for section in sections:
                        if section in policy_data['scores'] and policy_data['scores'][section]:
                            has_data = True
                            break
                    
                    if not has_data:
                        print(f"Warning: No evaluation data found for {policy_name}")
                        summaries[policy_name] = f"""**Current Compliance Status:**
No evaluation data found for {policy_name}. This could indicate that no applicable policy requirements were found or an error occurred during evaluation.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| No Evaluation Data | No applicable policy requirements found | Review policy mapping and evaluation process | Medium |"""
                        continue
                    
                    # Add timeout protection for the entire summary generation
                    try:
                        summary = await asyncio.wait_for(
                            generate_policy_summary(policy_name, policy_data, model_card_content, llm, sections),
                            timeout=600  # 10 minute timeout for entire summary generation
                        )
                        summaries[policy_name] = summary
                        print(f"Generated summary for {policy_name}")
                    except asyncio.TimeoutError:
                        print(f"Timeout error generating summary for {policy_name}")
                        summaries[policy_name] = f"""**Current Compliance Status:**
Timeout occurred while generating summary for {policy_name}. Please check the evaluation results manually.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| Summary Generation Timeout | Unable to generate detailed summary | Review evaluation results manually | Medium |"""
                    except Exception as e:
                        print(f"Error generating summary for {policy_name}: {str(e)}")
                        summaries[policy_name] = f"""**Current Compliance Status:**
Error occurred while generating summary for {policy_name}. Please check the evaluation results manually.

**Compliance Gaps and To-dos:**
| Compliance Gap | Description | To-dos | Priority |
|----------------|-------------|--------|----------|
| Summary Generation Error | {str(e)} | Review evaluation results manually | Medium |"""
                else:
                    print(f"Warning: No policy data found for {policy_name}")
            except Exception as e:
                print(f"Error processing policy file {policy_file}: {str(e)}")
                continue

        print("\nGenerated summaries for policies:", list(summaries.keys()))
        
        # Generate top-level summary
        print("\nGenerating top-level summary across all policies...")
        top_level_summary = await generate_top_level_summary(summaries, llm)
        print("Generated top-level summary")

        # Generate section-based summaries
        print("\nGenerating section-based summaries...")
        try:
            section_summaries = await generate_multiple_section_summaries(section_data, llm, TESTING_MODE)
            print(f"Generated summaries for {len(section_summaries)} sections")
        except Exception as e:
            print(f"Error generating section summaries: {str(e)}")
            # Fallback to individual processing if the main function fails
            section_summaries = {}
            for section in sections:
                try:
                    if not section_data[section]:  # Check if there's no data for this section
                        section_summaries[section] = json.dumps({
                            "Overall": f"""#### ⚠️ {section} – No Evaluation Data

Note: No evaluation data was provided for this section. This could indicate that:
- The section is missing from the model card
- No applicable policy requirements were found
- An error occurred during evaluation

Please ensure this section exists and contains the necessary information."""
                        })
                    else:
                        summary = await generate_section_summary(section, section_data[section], llm, TESTING_MODE)
                        section_summaries[section] = summary
                        print(f"Generated summary for section: {section}")
                except Exception as e:
                    print(f"Error generating summary for section {section}: {str(e)}")
                    section_summaries[section] = json.dumps({
                        "Overall": f"""#### ❌ {section} – Error

An error occurred while generating the summary for this section. Please check the logs for more details.""",
                        "Error": str(e)
                    })

        print("All evaluations completed.")
        return heatmap_filenames, summaries, top_level_summary, section_summaries
    except Exception as e:
        print(f"Error in AI pipeline: {str(e)}")
        raise