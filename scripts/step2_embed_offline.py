import json
from langchain_huggingface import HuggingFaceEmbeddings

INPUT_JSON = "data/chunks.json"
OUTPUT_EMBEDDINGS = "data/embeddings.json"

if __name__ == "__main__":
    print(f"Loading chunks from {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    texts = [c["page_content"] for c in chunks_data]
    print(f"Loaded {len(texts)} chunks.")

    print("\nLoading BAAI/bge-large-en-v1.5 embedding model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Embedding {len(texts)} documents. This might take a few minutes...")
    # Process in batches to avoid OOM
    vectors = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} (chunks {i} to {min(i+batch_size, len(texts))})...")
        batch_vectors = embeddings_model.embed_documents(batch)
        vectors.extend(batch_vectors)

    print(f"Successfully embedded {len(vectors)} chunks.")
    
    print(f"Saving embeddings to {OUTPUT_EMBEDDINGS}...")
    with open(OUTPUT_EMBEDDINGS, "w", encoding="utf-8") as f:
        json.dump(vectors, f)
    
    print("Done!")
