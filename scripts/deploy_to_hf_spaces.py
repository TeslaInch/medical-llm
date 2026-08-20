import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Error: HF_TOKEN not found in .env")
    exit(1)

api = HfApi(token=hf_token)

# The local folder containing our FastAPI app and Dockerfile
local_folder = "api"

# Get username
user_info = api.whoami()
username = user_info["name"]

# Space details
space_name = "Medical-RAG-API"
repo_id = f"{username}/{space_name}"

print(f"Creating HuggingFace Docker Space: {repo_id}...")
try:
    api.create_repo(
        repo_id=repo_id, 
        repo_type="space", 
        space_sdk="docker",
        private=False,
        exist_ok=True
    )
except Exception as e:
    print(f"Notice: {e}")

print(f"Uploading {local_folder} files to {repo_id}...")
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="space",
)

print(f"\n🚀 Success! Your API is deploying to: https://huggingface.co/spaces/{repo_id}")
print("Note: It may take a few minutes for HuggingFace to build the Docker image.")
