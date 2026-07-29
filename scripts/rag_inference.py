import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
import numpy as np
from sentence_transformers import CrossEncoder

# ── config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = "TeslaInch/scd-phi35-adapter-v2"  # v2 — our best model
VECTORDB_PATH = "/kaggle/input/scd-vectordb/scd_guidelines"
if not os.path.exists(VECTORDB_PATH):
    VECTORDB_PATH = "data/vectordb/scd_guidelines"

SYSTEM = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer clinical questions accurately using the provided guidelines. "
    "Always mention which guideline informed your answer."
)

# ── load model ────────────────────────────────────────────────────────────────
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    
    # Silence HuggingFace max_length warning
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# ── load vector database ──────────────────────────────────────────────────────
def load_vectordb():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages: "
    )
    return Chroma(
        persist_directory=VECTORDB_PATH,
        embedding_function=embeddings
    )

# ── RAG answer function ───────────────────────────────────────────────────────
def answer_with_rag(pipe, vectordb, cross_encoder, question, case=""):
    # retrieve relevant context
    search_query = f"{case} {question}" if case else question
    retrieved = vectordb.similarity_search(search_query, k=10) # fetch top 10

    # rerank
    if retrieved:
        pairs = [[search_query, doc.page_content] for doc in retrieved]
        scores = cross_encoder.predict(pairs)
        best_indices = np.argsort(scores)[::-1]
        
        # enforce source diversity
        seen_sources = set()
        diverse_indices = []
        for idx in best_indices:
            source = retrieved[idx].metadata.get("source", "unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                diverse_indices.append(idx)
            if len(diverse_indices) == 3:
                break
                
        # fallback if < 3 diverse sources exist
        if len(diverse_indices) < 3:
            for idx in best_indices:
                if idx not in diverse_indices:
                    diverse_indices.append(idx)
                if len(diverse_indices) == 3:
                    break
        
        # strategic placement: [best, third_best, second_best]
        if len(diverse_indices) >= 3:
            placed_indices = [diverse_indices[0], diverse_indices[2], diverse_indices[1]]
        else:
            placed_indices = diverse_indices
        final_docs = [retrieved[i] for i in placed_indices]
    else:
        final_docs = []

    context = "\n\n".join([
        f"[{doc.metadata.get('source', 'guideline')}]\n{doc.page_content}"
        for doc in final_docs
    ])

    # build augmented prompt
    clinical_content = f"Clinical case:\n{case}\n\nQuestion: {question}" if case else f"Question: {question}"

    user_content = f"""{SYSTEM}

RELEVANT CLINICAL GUIDELINES:
{context}

{clinical_content}

Answer using the guidelines above. Cite the source document."""

    prompt = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n"

    output = pipe(prompt, max_new_tokens=500, do_sample=False)
    response = output[0]["generated_text"].split("<|assistant|>")[-1].strip()
    response = response.replace("<|end|>", "").strip()

    sources = [doc.metadata.get("source", "unknown") for doc in final_docs]
    return response, sources

# ── test on your worst questions ──────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "question": "What is the hydroxyurea monitoring protocol for adults with sickle cell disease?",
        "case": ""
    },
    {
        "question": "What is the diagnosis and immediate management in priority order?",
        "case": "A 4-year-old boy with HbSS presents with fever of 39.2°C and swollen tender hands and feet bilaterally."
    },
    {
        "question": "What are the indications for exchange transfusion in sickle cell disease?",
        "case": ""
    },
    {
        "question": "What monitoring intervals are recommended for adults stabilising on hydroxyurea?",
        "case": ""
    },
    {
        "question": "What are the newborn screening recommendations for sickle cell disease in Nigeria?",
        "case": ""
    },
]

if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("Loading vector database...")
    vectordb = load_vectordb()
    print(f"Chunks loaded: {vectordb._collection.count()}")

    print("Loading Cross-Encoder Reranker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = CrossEncoder("BAAI/bge-reranker-large", device=device)

    print("\nRunning RAG inference tests...\n")
    results = []

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {q['question'][:60]}...")
        response, sources = answer_with_rag(
            pipe, vectordb, cross_encoder,
            q["question"],
            q["case"]
        )
        results.append({
            "question": q["question"],
            "case": q["case"],
            "response": response,
            "sources": sources
        })
        print(f"Sources: {sources}")
        print(f"Response: {response}\n")
        print("---")

    # save results
    with open("/kaggle/working/rag_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Results saved.") 