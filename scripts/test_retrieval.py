import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

OUTPUT_DIR = "data/vectordb/scd_guidelines"

if __name__ == "__main__":
    print("Loading BAAI/bge-large-en-v1.5 embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Connecting to Chroma database...")
    client = chromadb.PersistentClient(path=OUTPUT_DIR)
    collection = client.get_collection("langchain")

    test_queries = [
        "hydroxyurea monitoring protocol adults",
        "vaso-occlusive crisis immediate management",
        "sickle cell newborn screening Nigeria"
    ]

    print("\nTesting retrieval...")
    for query in test_queries:
        query_embedding = embeddings.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        print(f"\nQuery: {query}")
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            print(f"  Source: {meta.get('source')}")
            print(f"  Text: {doc[:150]}...\n")