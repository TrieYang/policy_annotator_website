import os
from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(model="claude-sonnet-4-20250514", anthropic_api_key=ANTHROPIC_API_KEY)

def extract_text_from_html(html_path):
    """Extracts all readable text from an HTML document while removing HTML elements."""
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Remove unnecessary elements
    for tag in soup(["script", "style", "meta", "link", "head", "footer", "nav", "aside"]):
        tag.decompose()

    extracted_text = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "li"]):
        extracted_text.append(tag.get_text(separator=" ").strip())

    return "\n\n".join(extracted_text)

def extract_relevant_recitals_annexes(policy_text):
    """Uses Claude AI to extract only Recitals & Annexes for policy compliance."""
    prompt = """
        You are a legal document analysis system designed to extract specific sections from policy documents with utmost accuracy. Your task is to extract the Recitals and Annexes from the following legal policy document, maintaining their exact wording and structure.

        Here is the policy document you need to analyze:

        {policy_text}

        Instructions:
        1. Read through the entire policy document carefully.
        2. Identify all Recitals and Annexes within the document.
        3. Extract these sections verbatim, preserving all text, formatting, and numbering exactly as they appear in the original document.
        4. Do not summarize, paraphrase, or modify any part of the extracted text.
        5. If a section is not present (e.g., no Recitals or no Annexes), explicitly state its absence.

        Before providing your final output, analyze the document inside <document_analysis> tags. This should include:
        1. Identifying and quoting the start and end of each Recital and Annex section.
        2. Counting the number of Recitals and Annexes found.
        3. Verifying that no text between the identified sections is missed.
        4. Confirming that you are extracting the full text without any modifications.
        5. Double-checking that the extracted content matches the original document exactly.

        It's OK for this section to be quite long.

        Your final output should be structured as follows:

        ---
        Recitals:
        [Full text of all Recitals, exactly as they appear in the document]

        Annexes:
        [Full text of all Annexes, exactly as they appear in the document]
        ---

        If either section is not present in the document, replace the [Full text...] with "No [Recitals/Annexes] found in the document."

        Example output structure (do not use this content, it's just to illustrate the format):

        ---
        Recitals:
        WHEREAS, Party A and Party B have entered into a contractual agreement dated [DATE];
        WHEREAS, Both parties wish to amend certain terms of the aforementioned agreement;
        ...
        [All other Recitals in full]

        Annexes:
        Annex 1: Definitions
        1.1 "Agreement" means...
        1.2 "Effective Date" refers to...
        ...
        [All other Annexes in full]
        ---

        Remember, your primary goal is to extract and present the exact text of the Recitals and Annexes without any alteration or summarization. Accuracy and completeness are paramount.
    """

    try:
        response = llm.invoke(prompt.format(policy_text=policy_text)).content
        print(response)
        return response
    except Exception as e:
        print(f"❌ Error extracting recitals & annexes: {e}")
        return ""

def save_extracted_text(extracted_text, output_folder):
    """Saves extracted recitals & annexes to separate text files."""
    os.makedirs(output_folder, exist_ok=True)

    # Splitting the response into Recitals and Annexes
    recitals_part = extracted_text.split("**Recitals:**")[1].split("**Annexes:**")[0].strip()
    annexes_part = extracted_text.split("**Annexes:**")[1].strip()

    recitals_file = os.path.join(output_folder, "recitals.txt")
    annexes_file = os.path.join(output_folder, "annexes.txt")

    with open(recitals_file, "w", encoding="utf-8") as f:
        f.write(recitals_part)
    print(f"✅ Recitals saved to {recitals_file}")

    with open(annexes_file, "w", encoding="utf-8") as f:
        f.write(annexes_part)
    print(f"✅ Annexes saved to {annexes_file}")

if __name__ == "__main__":
    HTML_FILE = "policy_docs/EU AI Act.html"
    OUTPUT_FOLDER = "policy_outputs"

    # Step 1: Extract raw text from HTML
    policy_text = extract_text_from_html(HTML_FILE)
    
    # Step 2: Use Claude to extract only relevant Recitals & Annexes
    extracted_text = extract_relevant_recitals_annexes(policy_text)

    # Step 3: Save the extracted text
    save_extracted_text(extracted_text, OUTPUT_FOLDER)



