# scripts/build_vectordb.py
import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("\nLoading BAAI/bge-large-en-v1.5 embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ── configuration ─────────────────────────────────────────────────────────────

# ── helper: clean text ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    
    in_references = False
    
    for line in lines:
        l_strip = line.strip()
        
        # Heuristic: stop processing if we hit References
        if re.match(r'^(?:\d+\.\s*)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*$', l_strip):
            in_references = True
            break
            
        if in_references:
            break
            
        # Skip empty lines or pure numbers (page numbers)
        if not l_strip or l_strip.isdigit():
            continue
            
        # Skip typical header/footer metadata patterns
        if re.search(r'(Copyright|Downloaded from|All rights reserved|DOI:|ISSN)', l_strip, re.IGNORECASE):
            continue
            
        # Skip lines that look like affiliations (very heuristic)
        if re.search(r'(University|Department of|Hospital|Institute|MD|PhD)', l_strip) and len(l_strip.split()) < 10:
            continue
            
        cleaned_lines.append(line)
        
    cleaned_text = "\n".join(cleaned_lines)
    # Fix broken words from hyphenation at end of lines
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    # Collapse multiple spaces
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    return cleaned_text

# ── load all PDFs ─────────────────────────────────────────────────────────────
PDF_DIR = "data/raw"
OUTPUT_DIR = "data/vectordb/scd_guidelines"

pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs:")
for f in pdf_files:
    print(f"  {f}")

all_docs = []
def process_pdf(pdf_path: str, pdf_file: str):
    """Extracts text from PDF, applying aggressive heuristics to remove noise."""
    import fitz # PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        
        cleaned_text = clean_text(full_text)
        
        if cleaned_text.strip():
            all_docs.append(Document(
                page_content=cleaned_text, 
                metadata={"source": pdf_file}
            ))
            print(f"  Processed {len(doc)} pages into clean document.")
        else:
            print(f"  Warning: No text remained after cleaning.")
            
    except Exception as e:
        print(f"  Error: {e} — skipping")

for pdf_file in pdf_files:
    path = os.path.join(PDF_DIR, pdf_file)
    print(f"\nLoading: {pdf_file}...")
    process_pdf(path, pdf_file)

print(f"\nTotal documents successfully cleaned and loaded: {len(all_docs)}")

# ── chunk documents ───────────────────────────────────────────────────────────
print("\nChunking documents...")
# Using larger chunk size to maintain context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# ── build vector database ─────────────────────────────────────────────────────
print("Building vector database in batches...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=OUTPUT_DIR
)

batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"  Embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} (chunks {i} to {min(i+batch_size, len(chunks))})...")
    vectordb.add_documents(batch)

print(f"\nVector database built successfully")
print(f"Total chunks indexed: {vectordb._collection.count()}")
print(f"Saved to: {OUTPUT_DIR}")

# ── test retrieval ────────────────────────────────────────────────────────────
print("\nTesting retrieval...")
test_queries = [
    "hydroxyurea monitoring protocol adults",
    "vaso-occlusive crisis immediate management",
    "sickle cell newborn screening Nigeria"
]

for query in test_queries:
    results = vectordb.similarity_search(query, k=2)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  Source: {r.metadata.get('source')}")
        print(f"  Text: {r.page_content[:150]}...\n")