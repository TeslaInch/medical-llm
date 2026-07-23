import sys
import json
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("Loading model...", file=sys.stderr)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Model loaded.", file=sys.stderr)
    
    # Signal readiness
    print("READY")
    sys.stdout.flush()
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        texts = json.loads(line)
        vectors = embeddings.embed_documents(texts)
        print(json.dumps(vectors))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
