"""
KAGGLE RAG RETRIEVAL EVALUATION PIPELINE
========================================
This script rigorously tests the RAG pipeline's Retrieval Quality (Context Precision and MRR)
using an LLM-as-a-judge approach. It evaluates whether the Vector DB actually retrieves
relevant medical context for complex clinical queries.

Instructions for Kaggle:
1. Ensure your Chroma DB is available in Kaggle (e.g., at `/kaggle/working/vectordb/scd_guidelines`).
2. Run in a cell: `!pip install langchain langchain-community chromadb sentence-transformers groq`
3. Add your Groq API Key below (or use Kaggle Secrets).
4. Run this script with GPU enabled (for fast query embedding).
"""

import os
import torch
from groq import Groq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
# Set your Groq API Key here (We use Llama-3-70b as the judge for fast, smart reasoning)
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# Path to the ChromaDB in your Kaggle environment
CHROMA_DB_DIR = "/kaggle/working/vectordb/scd_guidelines"

# A rigorous dataset of complex clinical questions
EVAL_QUERIES = [
    "What is the exact pediatric starting dose for hydroxyurea based on weight?",
    "Describe the emergency department triage flow chart for a vaso-occlusive crisis.",
    "What is the recommended laboratory safety monitoring schedule for a patient on hydroxyurea (CBC and reticulocyte count)?",
    "According to the SPARCo protocol, what are the specific criteria for hospitalization?",
    "What is the routine malaria chemoprophylaxis protocol in sickle cell disease for children under 5?",
    "At what age should transcranial doppler (TCD) screening begin for children with SCA?",
    "What are the specific indications for chronic red blood cell transfusion therapy?",
    "How should acute chest syndrome (ACS) be managed empirically with antibiotics?",
    "What is the target hemoglobin S (HbS) percentage for patients undergoing preoperative transfusion?",
    "What are the diagnostic criteria for avascular necrosis (AVN) of the femoral head in SCD?"
]

def judge_relevance(query: str, chunk_text: str, client: Groq) -> bool:
    """Uses Llama-3 as a judge to determine if the chunk contains the answer to the query."""
    prompt = f"""
    You are a strict medical evaluator. Your job is to determine if the provided medical document chunk 
    contains sufficient information to answer the user's query.
    
    Query: "{query}"
    
    Document Chunk:
    {chunk_text}
    
    Does the document chunk contain relevant information that directly helps answer the query?
    Respond with exactly one word: "YES" if it is relevant, or "NO" if it is irrelevant or just noise.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0,
            max_tokens=10,
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return False

def run_evaluation():
    if os.environ.get("GROQ_API_KEY") == "YOUR_GROQ_API_KEY":
        print("ERROR: Please set your GROQ_API_KEY before running!")
        return

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"ERROR: Chroma DB not found at {CHROMA_DB_DIR}. Make sure you built the DB first!")
        return

    print("Initializing Groq client...")
    groq_client = Groq()

    print("Loading BAAI/bge-large-en-v1.5 embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    embedding_func = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"Loading Chroma DB from {CHROMA_DB_DIR}...")
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_func)
    
    total_queries = len(EVAL_QUERIES)
    successful_queries = 0  # Queries where AT LEAST ONE relevant chunk was found
    mrr_sum = 0.0           # Mean Reciprocal Rank sum
    
    print("\nStarting LLM-as-a-Judge Retrieval Evaluation...\n")
    print("="*60)
    
    for idx, query in enumerate(EVAL_QUERIES, 1):
        print(f"Query {idx}/{total_queries}: {query}")
        
        # Retrieve top 5 chunks
        retrieved_docs = db.similarity_search(query, k=5)
        
        relevant_found = False
        first_relevant_rank = 0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            is_relevant = judge_relevance(query, doc.page_content, groq_client)
            
            if is_relevant:
                print(f"  [✓] Chunk {rank} is RELEVANT.")
                if not relevant_found:
                    first_relevant_rank = rank
                relevant_found = True
            else:
                print(f"  [X] Chunk {rank} is irrelevant.")
                
        if relevant_found:
            successful_queries += 1
            mrr_sum += (1.0 / first_relevant_rank)
            print(f"  --> Hit at Rank {first_relevant_rank}")
        else:
            print("  --> MISS (No relevant chunks found in Top 5)")
            
        print("-" * 60)

    # Calculate final metrics
    hit_rate = (successful_queries / total_queries) * 100
    mrr = mrr_sum / total_queries
    
    print("\n" + "="*60)
    print("FINAL RETRIEVAL METRICS")
    print("="*60)
    print(f"Total Queries Tested : {total_queries}")
    print(f"Hit Rate (Recall@5)  : {hit_rate:.2f}% (Queries where context was successfully found)")
    print(f"Mean Reciprocal Rank : {mrr:.4f} (1.0 means perfect ranking)")
    print("============================================================")
    
    if hit_rate >= 80 and mrr >= 0.7:
        print("\nCONCLUSION: Your RAG pipeline has EXCELLENT retrieval quality.")
        print("You can safely evaluate the LLMs knowing the context is solid.")
    else:
        print("\nCONCLUSION: Your RAG pipeline needs improvement before evaluating LLMs.")
        print("Check if the missed queries actually exist in the source documents.")

if __name__ == "__main__":
    run_evaluation()
