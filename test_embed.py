from langchain_community.embeddings import HuggingFaceEmbeddings
print("Loading model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Testing embed...")
embeddings.embed_documents(["Hello world"])
print("Success")
