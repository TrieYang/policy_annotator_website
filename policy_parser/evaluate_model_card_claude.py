import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Track costs and time (thread-safe)
start_time = time.time()
total_input_tokens = 0
total_output_tokens = 0
api_calls = 0
token_lock = threading.Lock()

# Pricing for Claude Sonnet 4
INPUT_COST_PER_1K = 0.003  # $3 per 1M tokens
OUTPUT_COST_PER_1K = 0.015  # $15 per 1M tokens

def calculate_cost(input_tokens, output_tokens):
    """Calculate API cost based on token usage."""
    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
    return input_cost + output_cost

# File paths
prompt_file = "prompt.txt"
legal_doc_file = "policy_outputs/EU_table.txt"
output_dir = "outputs"

# 5 Model cards to evaluate
model_cards = [
    ("Customer", "model_card_outputs/retail_customer_analytics_ai_model_card.txt"),
    ("Driving", "model_card_outputs/autonomous_driving_decision_engine_model_card.txt"),
    ("Insurance", "model_card_outputs/ai-powered_insurance_claim_evaluator_model_card.txt"),
    ("Medical", "model_card_outputs/ai-powered_medical_diagnosis_assistant_model_card.txt"),
    ("Resume", "model_card_outputs/ai_resume_screening_tool_model_card.txt")
]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load prompt template and legal document
with open(prompt_file, "r", encoding="utf-8") as f:
    prompt_template = f.read()
with open(legal_doc_file, "r", encoding="utf-8") as f:
    legal_doc_content = f.read()

def evaluate_model_card(model_name, model_card_path, index):
    """Evaluate a single model card against the legal document."""
    global total_input_tokens, total_output_tokens, api_calls
    
    # Create LLM instance for this thread
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.3
    )
    
    # Load model card
    with open(model_card_path, "r", encoding="utf-8") as f:
        model_card_content = f.read()
    
    # Create prompt - evaluate ALL sections (remove START/FINISH placeholders)
    prompt = (
        prompt_template
        .replace("{{MODEL_CARD}}", model_card_content)
        .replace("{{LEGAL_DOC}}", legal_doc_content)
    )
    
    # Remove the instruction about START/FINISH since we're evaluating all sections
    prompt = prompt.replace("- Only evaluate model card section from \"{{START}}\" to \"{{FINISH}}\".", "- Evaluate ALL sections in the model card.")
    
    try:
        call_start = time.time()
        response = llm.invoke(prompt, max_tokens=8192)
        call_duration = time.time() - call_start
        
        # Get actual token usage from API response
        input_tokens = 0
        output_tokens = 0
        
        # Try to get actual usage from response_metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('usage', {})
            if isinstance(usage, dict):
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
        
        # Fallback to estimation if not available
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = int(len(prompt.split()) * 1.3)
            output_tokens = int(len(response.content.split()) * 1.3)
        
        # Thread-safe token tracking
        with token_lock:
            api_calls += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            current_total_input = total_input_tokens
            current_total_output = total_output_tokens
        
        elapsed = time.time() - start_time
        current_cost = calculate_cost(current_total_input, current_total_output)
        print(f"   [{index}] {model_name}: ⏱️  {call_duration:.1f}s | Total: {elapsed/60:.1f}min | Cost: ${current_cost:.4f}")
        
        # Save output
        output_path = os.path.join(output_dir, f"evaluation_EU_{model_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.content)
        
        return {
            'model_name': model_name,
            'success': True,
            'output_path': output_path
        }
    except Exception as e:
        print(f"   [{index}] {model_name}: ❌ Error: {e}")
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }

print("=" * 70)
print("🚀 Starting EU AI Act Compliance Evaluation")
print("=" * 70)
print(f"📄 Legal Document: {legal_doc_file}")
print(f"📋 Model Cards: {len(model_cards)}")
print(f"💾 Output directory: {output_dir}")
print("=" * 70)

# Process all model cards concurrently
results = {}
with ThreadPoolExecutor(max_workers=len(model_cards)) as executor:
    # Submit all tasks
    future_to_model = {
        executor.submit(evaluate_model_card, name, path, i+1): (name, i+1)
        for i, (name, path) in enumerate(model_cards)
    }
    
    # Process completed tasks
    for future in as_completed(future_to_model):
        result = future.result()
        results[result['model_name']] = result

print("\n" + "=" * 70)
print("✅ Evaluation Complete!")
print("=" * 70)

# Calculate final totals
end_time = time.time()
elapsed_time = end_time - start_time
total_cost = calculate_cost(total_input_tokens, total_output_tokens)

# Save time and cost report
report_file = "policy_outputs/EU_evaluation_report.md"
report = f"""# EU AI Act Compliance Evaluation Report

## Processing Details
- **Start Time**: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
- **End Time**: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}
- **Total Processing Time**: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)
- **Model Cards Evaluated**: {len(model_cards)}

## API Usage
- **Total API Calls**: {api_calls}
- **Model Used**: claude-sonnet-4-20250514
- **Temperature**: 0.3

## Token Usage
- **Total Input Tokens**: {total_input_tokens:,.0f}
- **Total Output Tokens**: {total_output_tokens:,.0f}
- **Total Tokens**: {total_input_tokens + total_output_tokens:,.0f}

## Cost Estimation
- **Input Cost**: ${(total_input_tokens / 1000) * INPUT_COST_PER_1K:.4f}
- **Output Cost**: ${(total_output_tokens / 1000) * OUTPUT_COST_PER_1K:.4f}
- **Total Estimated Cost**: ${total_cost:.4f}

## Pricing Reference
- Input: ${INPUT_COST_PER_1K * 1000:.3f} per 1M tokens
- Output: ${OUTPUT_COST_PER_1K * 1000:.3f} per 1M tokens

## Notes
- Token counts are from actual API response when available, otherwise estimated
- All model cards processed concurrently for faster execution
- Evaluated against EU_table.txt (EU AI Act)
"""

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n📊 Evaluation report saved to: {report_file}")
print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
print(f"💰 Total estimated cost: ${total_cost:.4f}")
print("=" * 70)


