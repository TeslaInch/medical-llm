import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = "TeslaInch/scd-phi35-adapter-v2"  # v2 — our best model
VECTORDB_PATH = "/kaggle/input/scd-vectordb/scd_guidelines"

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

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# ── load vector database ──────────────────────────────────────────────────────
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=VECTORDB_PATH,
        embedding_function=embeddings
    )

# ── RAG answer function ───────────────────────────────────────────────────────
def answer_with_rag(pipe, vectordb, question, case=""):
    # retrieve relevant context
    search_query = f"{case} {question}" if case else question
    retrieved = vectordb.similarity_search(search_query, k=3)

    context = "\n\n".join([
        f"[{doc.metadata.get('source', 'guideline')}]\n{doc.page_content}"
        for doc in retrieved
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

    sources = [doc.metadata.get("source", "unknown") for doc in retrieved]
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
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("Loading vector database...")
    vectordb = load_vectordb()
    print(f"Chunks loaded: {vectordb._collection.count()}")

    print("\nRunning RAG inference tests...\n")
    results = []

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {q['question'][:60]}...")
        response, sources = answer_with_rag(
            pipe, vectordb,
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
        print(f"Response: {response[:300]}\n")
        print("---")

    # save results
    with open("/kaggle/working/rag_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Results saved.")