import json, time, os, requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
PDF_DIR = r"C:\Users\Gracetech\Desktop\medical-llm\data\raw"
OUTPUT_PATH = r"C:\Users\Gracetech\Desktop\medical-llm\data\train\scd_alpaca.json"

def generate_qa(chunk_text, source):
    prompt = f"""You are building a training dataset for a sickle cell disease AI.
Generate 3 clinical Q&A pairs from this guideline text.

Rules:
- Clinically realistic questions a doctor or medical student would ask
- Answers strictly from the provided text only
- Include specific numbers, drugs, thresholds, intervals where present
- Return ONLY a JSON array, no markdown, no explanation

Guideline text from {source}:
{chunk_text}

Return ONLY:
[
  {{"instruction": "question", "response": "answer"}},
  {{"instruction": "question", "response": "answer"}},
  {{"instruction": "question", "response": "answer"}}
]"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    })
    
    text = response.json()["response"].strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

# Load PDFs
print("Loading PDFs...")
all_docs = []
for f in os.listdir(PDF_DIR):
    if f.endswith(".pdf"):
        try:
            loader = PyPDFLoader(os.path.join(PDF_DIR, f))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = f
            all_docs.extend(docs)
            print(f"  ✓ {f} — {len(docs)} pages")
        except Exception as e:
            print(f"  ✗ {f} — {e}")

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(all_docs)

rich_chunks = [
    c for c in chunks
    if len(c.page_content.strip()) > 400
    and "reference" not in c.page_content.lower()[:50]
]
print(f"\n{len(rich_chunks)} chunks to process → ~{len(rich_chunks)*3} pairs expected")

# Resume if interrupted
all_pairs = []
start = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        existing = json.load(f)
    all_pairs = existing
    start = len(all_pairs) // 3
    print(f"Resuming from chunk {start} with {len(all_pairs)} existing pairs")

for i, chunk in enumerate(rich_chunks):
    if i < start:
        continue

    source = chunk.metadata.get("source", "guideline")

    try:
        pairs = generate_qa(chunk.page_content, source)
        for pair in pairs:
            if "instruction" in pair and "response" in pair:
                pair["source"] = source
                all_pairs.append(pair)

    except Exception as e:
        print(f"  Chunk {i} failed: {e}")
        continue

    if (i + 1) % 50 == 0:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_pairs, f, indent=2)
        print(f"  [{i+1}/{len(rich_chunks)}] {len(all_pairs)} pairs saved")

# Final save
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_pairs, f, indent=2)

print(f"\nDone — {len(all_pairs)} pairs saved to {OUTPUT_PATH}")

# Preview
for p in all_pairs[:2]:
    print(f"\nQ: {p['instruction']}")
    print(f"A: {p['response']}")