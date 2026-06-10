"""
push_train_to_hf.py
──────────────────
Pushes the compiled Instruction-Tuning dataset to HuggingFace.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from huggingface_hub import login

def main():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in .env")
        return

    login(token=hf_token)
    
    train_file = Path("data/train/instruction_dataset.json")
    if not train_file.exists():
        print(f"Error: {train_file} does not exist.")
        return

    print("Loading instruction dataset...")
    with open(train_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} training examples.")
    
    # Calculate 90/10 split
    split_idx = int(len(records) * 0.9)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records)
    })
    
    repo_id = "TeslaInch/SCD-Instruction-Tuning"
    print(f"\nPushing dataset to HuggingFace Hub: {repo_id}...")
    
    dataset_dict.push_to_hub(
        repo_id,
        private=False,
        commit_message="Initial upload of SCD instruction-tuning dataset."
    )
    
    print("\nSuccess! The training dataset has been published.")
    print(f"View it here: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
