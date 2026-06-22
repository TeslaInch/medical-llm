# scripts/test_retrieval.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

OUTPUT_DIR = "data/vectordb/scd_guidelines"

# check if database exists
if os.path.exists(OUTPUT_DIR):
    print(f"Vector database found at {OUTPUT_DIR}")
else:
    print("Database not found — need to rebuild")
    exit()

# load existing database
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Loading vector database...")
vectordb = Chroma(
    persist_directory=OUTPUT_DIR,
    embedding_function=embeddings
)

print(f"Chunks in database: {vectordb._collection.count()}")

# test retrieval
test_queries = [
    "hydroxyurea monitoring protocol adults",
    "vaso-occlusive crisis immediate management",
    "sickle cell newborn screening Nigeria",
    "acute chest syndrome treatment",
    "exchange transfusion indications"
]

print("\nTesting retrieval...")
for query in test_queries:
    results = vectordb.similarity_search(query, k=2)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  Source: {r.metadata.get('source')}")
        print(f"  Text: {r.page_content[:150]}...")