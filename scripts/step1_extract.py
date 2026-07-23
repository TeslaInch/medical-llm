import os
import re
import json
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = "data/raw"
OUTPUT_JSON = "data/chunks.json"

def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    in_references = False
    for line in lines:
        l_strip = line.strip()
        if re.match(r'^(?:\d+\.\s*)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*$', l_strip):
            in_references = True
            break
        if in_references:
            break
        if not l_strip or l_strip.isdigit():
            continue
        if re.search(r'(Copyright|Downloaded from|All rights reserved|DOI:|ISSN)', l_strip, re.IGNORECASE):
            continue
        if re.search(r'(University|Department of|Hospital|Institute|MD|PhD)', l_strip) and len(l_strip.split()) < 10:
            continue
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    return cleaned_text

def process_pdf(pdf_path: str, pdf_file: str) -> Document:
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        cleaned_text = clean_text(full_text)
        if cleaned_text.strip():
            print(f"  Processed {len(doc)} pages into clean document.")
            return Document(page_content=cleaned_text, metadata={"source": pdf_file})
        else:
            print(f"  Warning: No text remained after cleaning.")
            return None
    except Exception as e:
        print(f"  Error: {e} — skipping")
        return None

if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs:")
    for f in pdf_files:
        print(f"  {f}")

    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        print(f"\nLoading: {pdf_file}...")
        doc = process_pdf(path, pdf_file)
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
