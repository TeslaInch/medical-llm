import os
import re
import json
import nest_asyncio
from llama_parse import LlamaParse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Required for LlamaParse async loop management in some environments
nest_asyncio.apply()

PDF_DIR = "data/raw"
OUTPUT_JSON = "data/chunks.json"

def clean_markdown(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    in_references = False
    
    for line in lines:
        l_strip = line.strip()
        
        # Toggle references ON - looks for "References" or "Bibliography"
        if re.match(r'^(?:#+\s*)?(?:\d+\.\s*)?(References|Bibliography)\s*$', l_strip, re.IGNORECASE):
            in_references = True
            
        # Toggle references OFF - scan forward for 'Appendix' or 'Annex' and resume extraction.
        if in_references and re.search(r'(?i)\b(Appendix|Annex)\b', l_strip):
            in_references = False
            
        if in_references:
            continue
            
        if not l_strip or l_strip.isdigit():
            continue
        if re.search(r'(Copyright|Downloaded from|All rights reserved|DOI:|ISSN)', l_strip, re.IGNORECASE):
            continue
            
        # Strip simple affiliations but carefully keep table markdown (which contains '|')
        if not "|" in l_strip and re.search(r'(University|Department of|Hospital|Institute|MD|PhD)', l_strip) and len(l_strip.split()) < 10:
            continue
            
        cleaned_lines.append(line)
        
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    return cleaned_text

def process_pdf(pdf_path: str, pdf_file: str, parser: LlamaParse) -> Document:
    try:
        print(f"\nProcessing {pdf_file} via LlamaParse...")
        # LlamaParse natively extracts images, tables, and graphs as Markdown
        parsed_docs = parser.load_data(pdf_path)
        
        # Combine pages into a single text block for contiguous cleaning
        full_text = "\n\n".join([d.text for d in parsed_docs])
        cleaned_text = clean_markdown(full_text)
        
        if cleaned_text.strip():
            print(f"  Successfully processed and cleaned {pdf_file}")
            return Document(page_content=cleaned_text, metadata={"source": pdf_file})
        else:
            print(f"  Warning: No text remained after cleaning {pdf_file}")
            return None
            
    except Exception as e:
        print(f"  Error processing {pdf_file}: {e}")
        return None

if __name__ == "__main__":
    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        print("WARNING: LLAMA_CLOUD_API_KEY is not set in the environment variables.")
        print("Please set it before running, otherwise LlamaParse will fail.")

    print("Initializing LlamaParse (Multimodal enabled)...")
    parser = LlamaParse(
        result_type="markdown",
        vendor_multimodal_model=True,  # Handles images and flowcharts natively
        verbose=True
    )

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs:")
    for f in pdf_files:
        print(f"  {f}")

    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        doc = process_pdf(path, pdf_file, parser)
        if doc:
            all_docs.append(doc)

    print(f"\nTotal documents successfully cleaned and loaded: {len(all_docs)}")

    print("\nChunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")
    
    # Save to JSON
    chunks_data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_JSON}")
