from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse

predefined_urls = [
    "https://gdpr-info.eu",
    "https://w3id.org/GDPRtEXT/",
    "https://privonto.org/",
    "https://zenodo.org/record/5139467",
    "https://www.usableprivacy.org/data",
    "https://huggingface.co/nlpaueb/legal-bert-base-uncased",
    "https://data.europa.eu/en"
]

def change_name(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    safe_name = domain.replace("/", "_").replace(":", "_")
    return safe_name

def save_text_to_file(text: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved extracted content to: {filename}")

def main():
    print("Select a URL to load:\n")
    for i, url in enumerate(predefined_urls, start=1):
        print(f"{i}. {url}")
    
    choice = input("\nEnter the number of the URL you want to load: ").strip()
    
    if not choice.isdigit() or not (1 <= int(choice) <= len(predefined_urls)):
        print("Invalid choice. Please enter a valid number.")
        return

    selected_url = predefined_urls[int(choice) - 1]
    print(f"\nLoading content from: {selected_url}\n")

    try:
        loader = WebBaseLoader(selected_url)
        data = loader.load()
        print(f"Successfully loaded {len(data)} document(s) from {selected_url}\n")
        
        if not data:
            print("No content extracted from this URL.")
            return
        
        full_text = "\n\n".join([doc.page_content for doc in data])
        
        text_preview = full_text[:1000].replace("\n", " ")
        print("--- Document Preview (first 1000 characters) ---")
        print(text_preview)
        print("\n----------------------------------------------------")
        
        filename = f"{change_name(selected_url)}_data.txt"
        save_text_to_file(full_text, filename)
        
    except Exception as e:
        print(f"Error loading content: {e}")

if __name__ == "__main__":
    main()
