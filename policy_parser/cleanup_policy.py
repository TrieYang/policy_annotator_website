import os
import time
import pandas as pd
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(model="claude-sonnet-4-20250514", anthropic_api_key=ANTHROPIC_API_KEY)

def load_reference_documents(main_folder):

    main_articles_text = ""
    for filename in sorted(os.listdir(main_folder)): 
        if filename.endswith(".txt"):
            with open(os.path.join(main_folder, filename), "r", encoding="utf-8") as f:
                main_articles_text += f"### {filename} ###\n{f.read().strip()}\n\n"

    return main_articles_text

def extract_policy_clauses(article_text):
    prompt = f"""
    #You will be provided with
        1, An article of a Policy Document (Current Articles for Processing), structured in markdown table format

    #Instruction
        1, For each row, the first column is the article number, the second column is the policy clause number(within article), the third column is the policy clause content.
        2, Remove the row if the policy clause content falls into the following categories: 
            a) Purpose of Document Clauses: These clauses provide statements about the objective of the regulation. Example: "The purpose of this Regulation is to improve the functioning of the internal market and promote the uptake of human-centric and trustworthy artificial intelligence..."
            b) Any regulation Enforcement and administrative clauses, for example: 
                - Administrative and Institutional Clauses: These describe the role of regulatory bodies, reporting structures, and procedures. Example: "A European Artificial Intelligence Board (the 'Board') is hereby established."
                - Anything Related to Execution of Policy, for example: "The power to adopt delegated acts is conferred on the Commission subject to the conditions laid down in this Article."
                - Anything that refers to "the commission" or "fine".
                - Requirements/Direction for Regulatory Authorities: These clauses focus on how regulators should enforce the law. Example: "Upon such qualified alert, the Commission, through the AI Office and after having informed the Board, may exercise the powers laid down in this Section for the purpose of assessing the matter", "The Commission shall be assisted by a committee. That committee shall be a committee within the meaning of Regulation (EU) No 182/2011."
                - Rules on International Cooperation and Member State Responsibilities: These cover the collaboration between EU member states and external entities. Example: "Member States may call upon experts of the scientific panel to support their enforcement activities under this Regulation."
                - Market Surveillance and Certification Clauses: These describe how AI systems are certified and monitored by external authorities. Example: "Certificates issued by notified bodies shall be drawn-up in a language which can be easily understood by the relevant authorities..." 
                - Policy plannning: forward-looking policy planning statements for regulators. Example: "The Commission shall, if necessary, submit appropriate proposals to amend this Regulation, in particular taking into account developments in technology, the effect of AI systems on health and safety..."
                - Execution of policy: Clauses to direct regulator's judgement and behavior, for example: "When deciding whether to impose an administrative fine and when deciding on the amount of the administrative fine in each individual case, all relevant circumstances of the specific situation shall be taken into account and, as appropriate..."
                - Non-compliance consequences: describes the consequences(usually it talks about fines and procedures) for non-compliance, for example: "The non-compliance of the AI system with any requirements or obligations under this Regulation, other than those laid down in Article 5, shall be subject to administrative fines of up to EUR 750 000", "Non-compliance with any of the following provisions related to operators or notified bodies, other than those laid down in Articles 5, shall be subject to administrative fines of up to EUR 15 000 000"
        3, do not modify any other rows, do not change the number in column 2.

    #A section of Policy Document (Current Article for Processing)**
        {article_text}


    # IMPORTANT: I want you to continue with your response even when the output is lengthy. Your response should be only consisted of the markdown table, no extra question/note/description should be output.
    """

    try:
        response = llm.invoke(prompt, max_tokens=8192).content
        print(response)
        return response
    except Exception as e:
        print(f"❌ Error extracting policy clauses: {e}")
        return ""

def append_extracted_content(extracted_text, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Split text into rows
    lines = extracted_text.strip().split("\n")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Appended content to: {output_file}")

# === 4. MAIN FUNCTION TO PROCESS ALL POLICY ARTICLES ===
if __name__ == "__main__":
    MAIN_FOLDER = "policy_outputs/Articles_structured"  
    OUTPUT_MD_FILE = "policy_outputs/cleanup.md"  
    OUTPUT_TXT_FILE = "policy_outputs/cleanup.txt" 
    main_articles_text = load_reference_documents(MAIN_FOLDER)

    print(os.listdir(MAIN_FOLDER))


    # Clear existing output files before writing new content
    open(OUTPUT_MD_FILE, "w").close() 
    open(OUTPUT_TXT_FILE, "w").close() 
    

    for filename in sorted(
    [f for f in os.listdir(MAIN_FOLDER) if f.startswith("Article_") and f.endswith(".txt")],  
    key=lambda x: int(x.split('_')[1].split('.')[0])
):
        if filename.endswith(".txt"):
            time.sleep(10)
            article_path = os.path.join(MAIN_FOLDER, filename)
            print(f"📄 Processing: {filename}...")

            with open(article_path, "r", encoding="utf-8") as f:
                article_text = f.read().strip()
            extracted_text = extract_policy_clauses(article_text)

            append_extracted_content(extracted_text, OUTPUT_MD_FILE)
            append_extracted_content(extracted_text, OUTPUT_TXT_FILE)