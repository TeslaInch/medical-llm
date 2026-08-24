import os
import subprocess
import shutil
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Error: HF_TOKEN not found in .env")
    exit(1)

api = HfApi(token=hf_token)

# Build React Frontend
print("Building React Frontend...")
try:
    # Run npm run build in frontend directory (windows syntax handled by shell=True)
    subprocess.run(["npm", "run", "build"], cwd="frontend", check=True, shell=True)
    print("React build successful.")
    
    # Copy dist folder to api folder
    api_dist_path = os.path.join("api", "dist")
    frontend_dist_path = os.path.join("frontend", "dist")
    
    if os.path.exists(api_dist_path):
        shutil.rmtree(api_dist_path)
    shutil.copytree(frontend_dist_path, api_dist_path)
    print(f"Copied frontend build to {api_dist_path}")
    
except Exception as e:
    print(f"Error building frontend: {e}")
    exit(1)

# The local folder containing our FastAPI app and Dockerfile
local_folder = "api"

# Get username
user_info = api.whoami()
username = user_info["name"]

# Space details
space_name = "SCD-Medical-LLM"
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
