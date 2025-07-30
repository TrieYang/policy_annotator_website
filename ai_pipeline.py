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
from utils.section_summary import generate_section_summary
from utils.top_level_summary import generate_top_level_summary
from utils.interactive_heatmap import generate_interactive_heatmap
from utils.policy_summary import generate_policy_summary

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
        prompt_template_path = "new_prompt.txt"
        async with aiofiles.open(prompt_template_path, "r", encoding="utf-8") as f:
            chunk_prompt = await f.read()
            
        # Read prompt template
        prompt_template_path = "new_prompt_second.txt"
        async with aiofiles.open(prompt_template_path, "r", encoding="utf-8") as f:
            chunk_prompt_temp = await f.read()

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
      
                # Load the relevancy map for the current policy
                section_chunks = load_relevancy_map(policy_file.split('.')[0])

                # Initialize section data for this policy
                policy_section_scores = {section: {} for section in sections}
                policy_section_descriptions = {section: {} for section in sections}

                for section in sections:
                        # Get the chunks for this specific section
                        section_specific_chunks = section_chunks.get(section, [])
                        if not section_specific_chunks:
                            print(f"No chunks for section '{section}' - skipping evaluation")
                            continue
                            
                        for chunk in section_specific_chunks:
                            # Convert the chunk of articles into a comma-separated string
                            articles_str = ", ".join(chunk)
                            chunk_prompt_second = (
                                chunk_prompt_temp
                                .replace("{{LEGAL_DOC}}", legal_doc_content)
                                .replace("{{SECTION}}", section)
                                .replace("{{ARTICLES_TO_EVALUATE}}", articles_str)
                            )
                            print(f"Evaluating {policy_file} section '{section}' for articles {articles_str}...")
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
                                            "text": f"{chunk_prompt}",
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
                                    "content": f"{chunk_prompt_second}",
                                },
                            ]
                            response = llm.invoke(messages)
                            print(response)
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
                                    except (ValueError, json.JSONDecodeError) as e:
                                        print(f"Error parsing row {data_row}: {e}")
                                        continue
                            except Exception as e:
                                print("something went wrong")
                                print(f"Error processing response: {e}")
                                print(f"Full response:\n{response}")
                           

                # Store the data for this policy
                policy_name = policy_file.split('.')[0]
                
                # Get all article numbers and convert to float for proper sorting
                all_articles = set()
                for section in sections:
                    all_articles.update(policy_section_scores[section].keys())
                
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
                    descriptions_df.loc[section, column] = policy_data['descriptions'][section].get(article_key, "No evaluation")
        
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
                    summary = await generate_policy_summary(policy_name, policy_data, model_card_content, llm, sections)
                    summaries[policy_name] = summary
                    print(f"Generated summary for {policy_name}")
            except Exception as e:
                print(f"Error generating summary for policy {policy_name}: {str(e)}")
                continue

        print("\nGenerated summaries for policies:", list(summaries.keys()))
        
        # Generate top-level summary
        print("\nGenerating top-level summary across all policies...")
        top_level_summary = await generate_top_level_summary(summaries, llm)
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
                    summary = await generate_section_summary(section, section_data[section], llm, TESTING_MODE)
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