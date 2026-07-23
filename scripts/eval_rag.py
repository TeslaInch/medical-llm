import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import json

OUTPUT_DIR = "data/vectordb/scd_guidelines"

queries = [
    "Emergency department triage flow chart for vaso-occlusive crisis",
    "Pediatric hydroxyurea weight-based initial dosing table",
    "Laboratory safety monitoring schedule for hydroxyurea CBC reticulocyte count",
    "SPARCo protocol",
    "Routine malaria chemoprophylaxis protocol in sickle cell disease"
]

def main():
    print("Loading BAAI/bge-large-en-v1.5 embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Connecting to Chroma database...")
    try:
        client = chromadb.PersistentClient(path=OUTPUT_DIR)
        collection = client.get_collection("langchain")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Running evaluation queries...\n")
    for q in queries:
        print("="*80)
        print(f"QUERY: {q}")
        print("="*80)
        
        query_embedding = embeddings.embed_query(q)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            
            print(f"--- Result {i+1} ---")
            print(f"Source: {meta.get('source')} | Distance: {dist:.4f}")
            # Print first 200 chars to avoid flooding the terminal
            print(doc[:300] + "...\n")

if __name__ == "__main__":
    main()
