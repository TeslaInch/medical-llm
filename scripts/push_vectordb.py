import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Error: HF_TOKEN not found in .env")
    exit(1)

api = HfApi(token=hf_token)

# The local folder containing Chroma DB
local_folder = "data/vectordb/scd_guidelines"

# HuggingFace repository details
# Getting the username from the token
user_info = api.whoami()
username = user_info["name"]

# Repo names cannot have spaces, so we use dashes
repo_name = "SCD-vectorDB-v1"
repo_id = f"{username}/{repo_name}"

print(f"Creating dataset repository: {repo_id}...")
try:
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
except Exception as e:
    print(f"Repo creation notice: {e}")

print(f"Uploading {local_folder} to {repo_id}...")
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",
)

print(f"\nSuccess! VectorDB pushed to: https://huggingface.co/datasets/{repo_id}")
