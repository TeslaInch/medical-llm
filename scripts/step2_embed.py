import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

OUTPUT_JSON = "data/chunks.json"
OUTPUT_DIR = "data/vectordb/scd_guidelines"

if __name__ == "__main__":
    print(f"Loading chunks from {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks_data]
    print(f"Loaded {len(chunks)} chunks.")

    print("\nLoading BAAI/bge-large-en-v1.5 embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

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
