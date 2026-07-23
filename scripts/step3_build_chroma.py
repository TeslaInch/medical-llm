import os
import json
import chromadb

INPUT_CHUNKS = "data/chunks.json"
INPUT_EMBEDDINGS = "data/embeddings.json"
OUTPUT_DIR = "data/vectordb/scd_guidelines"

if __name__ == "__main__":
    print(f"Loading chunks from {INPUT_CHUNKS}...")
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    print(f"Loading embeddings from {INPUT_EMBEDDINGS}...")
    with open(INPUT_EMBEDDINGS, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    if len(chunks_data) != len(embeddings_data):
        raise ValueError(f"Mismatch: {len(chunks_data)} chunks vs {len(embeddings_data)} embeddings.")

    print("\nBuilding vector database in batches...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    client = chromadb.PersistentClient(path=OUTPUT_DIR)
    collection = client.get_or_create_collection("langchain")

    batch_size = 50
    for i in range(0, len(chunks_data), batch_size):
        chunk_batch = chunks_data[i:i + batch_size]
        emb_batch = embeddings_data[i:i + batch_size]
        
        texts = [c["page_content"] for c in chunk_batch]
        metadatas = [c["metadata"] for c in chunk_batch]
        ids = [f"doc_{i+j}" for j in range(len(chunk_batch))]
        
        print(f"  Inserting batch {i//batch_size + 1}/{(len(chunks_data) + batch_size - 1)//batch_size}...")
        collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=emb_batch,
            ids=ids
        )

    print(f"\nVector database built successfully")
    print(f"Total chunks indexed: {collection.count()}")
    print(f"Saved to: {OUTPUT_DIR}")
