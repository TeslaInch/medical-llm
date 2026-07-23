import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import sys

OUTPUT_DIR = "data/vectordb/scd_guidelines"

def main():
    print("==================================================")
    print("   SCD RAG Interactive Tester")
    print("==================================================")
    print("Loading BAAI/bge-large-en-v1.5 embedding model (takes ~1-2 mins)...")
    
    # We load the embedding model to encode the user's live queries
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
        print(f"Failed to connect to Vector DB: {e}")
        return
    
    total_docs = collection.count()
    print(f"Database connected! Total chunks in DB: {total_docs}")
    print("Type 'quit' or 'exit' to stop.")
    print("==================================================\n")

    while True:
        try:
            query = input("\nEnter your test query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting tester...")
                break
            if not query:
                continue

            k_input = input("How many chunks to retrieve? (default 3): ").strip()
            k = int(k_input) if k_input.isdigit() else 3

            print(f"\nSearching for top {k} results...\n")
            
            query_embedding = embeddings.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                print("No results found.")
                continue

            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i] if results["distances"] else "N/A"
                
                print("-" * 80)
                print(f"RESULT #{i+1} | Source: {meta.get('source', 'Unknown')} | Distance: {dist}")
                print("-" * 80)
                # We print the raw chunk EXACTLY as it appears in the database
                print(doc)
                print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\nExiting tester...")
            break
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    main()
