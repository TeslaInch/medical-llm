import os
import json
import subprocess
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

class RemoteEmbeddings(Embeddings):
    def __init__(self):
        self.proc = subprocess.Popen(
            ["python", "scripts/embed_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Wait for ready signal
        while True:
            line = self.proc.stdout.readline().strip()
            if line == "READY":
                break

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.proc.stdin.write(json.dumps(texts) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)
        
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
        
    def close(self):
        self.proc.terminate()

OUTPUT_JSON = "data/chunks.json"
OUTPUT_DIR = "data/vectordb/scd_guidelines"

if __name__ == "__main__":
    print(f"Loading chunks from {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks_data]
    print(f"Loaded {len(chunks)} chunks.")

    print("\nStarting remote embedding server...")
    embeddings = RemoteEmbeddings()

    print("Building vector database in batches...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=OUTPUT_DIR
    )

    batch_size = 50
    try:
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
    finally:
        embeddings.close()
