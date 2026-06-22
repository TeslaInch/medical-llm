# scripts/build_vectordb.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── load all PDFs ─────────────────────────────────────────────────────────────
PDF_DIR = "data/raw"
OUTPUT_DIR = "data/vectordb/scd_guidelines"

pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs:")
for f in pdf_files:
    print(f"  {f}")

all_docs = []
for pdf_file in pdf_files:
    path = os.path.join(PDF_DIR, pdf_file)
    print(f"\nLoading: {pdf_file}...")
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # tag each chunk with its source filename
        for doc in docs:
            doc.metadata["source"] = pdf_file
        all_docs.extend(docs)
        print(f"  Pages loaded: {len(docs)}")
    except Exception as e:
        print(f"  Error: {e} — skipping")

print(f"\nTotal pages loaded: {len(all_docs)}")

# ── chunk documents ───────────────────────────────────────────────────────────
print("\nChunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# ── build vector database ─────────────────────────────────────────────────────
print("\nLoading embedding model (first run downloads ~400MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Building vector database...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=OUTPUT_DIR
)

print(f"\nVector database built successfully")
print(f"Total chunks indexed: {vectordb._collection.count()}")
print(f"Saved to: {OUTPUT_DIR}")

# ── test retrieval ────────────────────────────────────────────────────────────
print("\nTesting retrieval...")
test_queries = [
    "hydroxyurea monitoring protocol adults",
    "vaso-occlusive crisis immediate management",
    "sickle cell newborn screening Nigeria",
    "acute chest syndrome treatment",
    "exchange transfusion indications"
]

for query in test_queries:
    results = vectordb.similarity_search(query, k=2)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  Source: {r.metadata.get('source')}")
        print(f"  Text: {r.page_content[:150]}...")