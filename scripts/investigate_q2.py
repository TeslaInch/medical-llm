import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder
import numpy as np

OUTPUT_DIR = "data/vectordb/scd_guidelines"

def main():
    print("Loading embedding model...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages: "
    )
    
    print("Loading Chroma DB...")
    client = chromadb.PersistentClient(path=OUTPUT_DIR)
    collection = client.get_collection("langchain")
    
    # 1. Direct search for Dactylitis
    print("\n--- Direct Search for 'dactylitis' / 'hand-foot syndrome' ---")
    dact_emb = embeddings.embed_query("dactylitis hand-foot syndrome swollen tender hands and feet")
    dact_results = collection.query(
        query_embeddings=[dact_emb],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    for i in range(len(dact_results["documents"][0])):
        print(f"Source: {dact_results['metadatas'][0][i].get('source')}, Distance: {dact_results['distances'][0][i]}")
        print(f"Content: {dact_results['documents'][0][i][:300]}...\n")
        
    # 2. Retrieval for Question 2
    print("\n--- Retrieval for Question 2 ---")
    question = "What is the diagnosis and immediate management in priority order?"
    case = "A 4-year-old boy with HbSS presents with fever of 39.2°C and swollen tender hands and feet bilaterally."
    search_query = f"{case} {question}"
    
    q2_emb = embeddings.embed_query(search_query)
    q2_results = collection.query(
        query_embeddings=[q2_emb],
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )
    
    print("Loading Reranker...")
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    
    pairs = [[search_query, doc] for doc in q2_results["documents"][0]]
    scores = cross_encoder.predict(pairs)
    best_indices = np.argsort(scores)[::-1]
    
    print("\nTop 5 Reranked Chunks for Q2:")
    for rank, idx in enumerate(best_indices[:5]):
        doc = q2_results["documents"][0][idx]
        meta = q2_results["metadatas"][0][idx]
        print(f"\n[Rank {rank+1}] Score: {scores[idx]:.4f} | Source: {meta.get('source')}")
        
        has_dactylitis = "dactylitis" in doc.lower()
        has_sepsis = "sepsis" in doc.lower()
        has_fever = "fever" in doc.lower()
        print(f"Contains -> Dactylitis: {has_dactylitis}, Sepsis: {has_sepsis}, Fever: {has_fever}")
        print("-" * 50)
        print(doc)
        print("=" * 80)

if __name__ == "__main__":
    main()
