import os
from bs4 import BeautifulSoup

def clean_html(html_path, output_folder):
    """
    Removes HTML, CSS, and JavaScript elements from an HTML file and extracts only readable text.
    Saves the cleaned text into a .txt file.
    """
    # Read the HTML file
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Remove all non-text elements
    for tag in soup(["script", "style", "meta", "link", "head", "footer", "nav", "aside"]):
        tag.decompose()

    # Extract readable content
    extracted_text = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "li", "span"]):
        extracted_text.append(tag.get_text(separator=" ").strip())

    # Join extracted text into one document
    cleaned_text = "\n\n".join(extracted_text)

    # Save cleaned text as a .txt file
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(html_path))[0] + "_cleaned.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ Cleaned text saved to: {output_file}")

if __name__ == "__main__":
    HTML_FILE = "policy_docs/EU AI Act.html" 
    OUTPUT_FOLDER = "policy_outputs"

    clean_html(HTML_FILE, OUTPUT_FOLDER)
