"""
KAGGLE RAG PIPELINE WITH DOCLING (FIXED)
=========================================
"""

import os
import re
import torch
import asyncio
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
PDF_DIR = "/kaggle/input/datasets/teslaincarnate/scd-pdfs"
OUTPUT_DIR = "/kaggle/working"
CHROMA_DB_DIR = os.path.join(OUTPUT_DIR, "vectordb", "scd_guidelines")

def clean_markdown(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    in_references = False
    
    for line in lines:
        l_strip = line.strip()
        
        is_table_line = "|" in l_strip
        
        if re.match(r'^(?:#+\s*)?(?:\d+\.\s*)?(References|Bibliography)\s*$', l_strip, re.IGNORECASE):
            in_references = True
            
        if in_references and re.search(r'(?i)\b(Appendix|Annex)\b', l_strip):
            in_references = False
            
        if in_references:
            continue
            
        if is_table_line:
            cleaned_lines.append(line)
            continue

        if not l_strip:
            continue
            
        if l_strip.isdigit() and len(l_strip) <= 3:
            continue
            
        if re.search(r'(Copyright|Downloaded from|All rights reserved|DOI:|ISSN)', l_strip, re.IGNORECASE):
            continue
            
        if re.search(r'^(Author|Affiliations|Correspondence to:)', l_strip, re.IGNORECASE):
            continue
            
        cleaned_lines.append(line)
        
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    return cleaned_text

async def run_pipeline():
    if not os.path.exists(PDF_DIR):
        print(f"ERROR: PDF_DIR '{PDF_DIR}' not found.")
        return

    print("Initializing Docling Document Converter...")
    converter = DocumentConverter()
    
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDFs.\n")
    
    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        try:
            # Use convert() instead of convert_single()
            doc_result = converter.convert(path)
            
            # Extract markdown from the document result
            full_text = doc_result.document.export_to_markdown()
            cleaned_text = clean_markdown(full_text)
            
            if cleaned_text.strip():
                all_docs.append(Document(
                    page_content=cleaned_text, 
                    metadata={"source": pdf_file}
                ))
                print(f"  ✓ Successfully processed {pdf_file}")
            else:
                print(f"  ⚠ Warning: No text remained after cleaning {pdf_file}")
                
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Total documents loaded: {len(all_docs)}")
    print(f"{'='*60}\n")

    if len(all_docs) == 0:
        print("ERROR: No documents were parsed. Aborting.")
        return

    print("Chunking documents with Markdown/Table awareness...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=250,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n|", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks generated: {len(chunks)}")
    
    if len(chunks) == 0:
        print("ERROR: Splitting failed, 0 chunks generated. Aborting.")
        return

    print("\nLoading BAAI/bge-large-en-v1.5 embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    embedding_func = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("\nBuilding Chroma Vector DB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_func,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"\n✓ Pipeline complete! Chroma DB persisted to: {CHROMA_DB_DIR}")

# Run the pipeline
await run_pipeline()