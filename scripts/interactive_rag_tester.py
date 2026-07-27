import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import sys

OUTPUT_DIR = "data/vectordb/scd_guidelines"

def main():
    print("Loading BAAI/bge-large-en-v1.5 embedding model...")
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: "
        )
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return

    print(f"Connecting to Chroma database at {OUTPUT_DIR}...")
    try:
        client = chromadb.PersistentClient(path=OUTPUT_DIR)
        collection = client.get_collection("langchain")
        count = collection.count()
        print(f"Successfully connected! Database contains {count} document chunks.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("\n" + "="*80)
    print(" Interactive RAG Tester ".center(80, "="))
    print("="*80)
    print("Query the database deeply to find bugs, metadata leaks, or reference artifacts.")
    print("Distance < 0.35 is excellent, > 0.55 is poor.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*80 + "\n")

    while True:
        try:
            query = input("\nEnter your test query > ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if query.strip().lower() in ['exit', 'quit']:
            break
            
        if not query.strip():
            continue
            
        print("\nSearching...")
        try:
            query_embedding = embeddings.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5, # Get top 5 to be more comprehensive
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                print("No results found.")
                continue
                
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                
                print(f"\n" + f" RESULT {i+1} ".center(80, "-"))
                print(f"Distance Score : {dist:.4f}")
                print(f"Source File    : {meta.get('source', 'Unknown')}")
                print(f"Page / Loc     : {meta.get('page', 'Unknown')}")
                print(f"Full Metadata  : {meta}")
                print("-" * 80)
                print(doc)
                print("-" * 80)
                
        except Exception as e:
            print(f"Error during search: {e}")

if __name__ == "__main__":
    main()
