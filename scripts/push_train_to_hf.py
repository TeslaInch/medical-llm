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
    
    train_file = Path("data/train/scd_training_final.json")
    if not train_file.exists():
        print(f"Error: {train_file} does not exist.")
        return

    print("Loading instruction dataset...")
    with open(train_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    train_records = records.get("train", [])
    val_records = records.get("validation", [])
    print(f"Loaded {len(train_records)} training and {len(val_records)} validation examples.")
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records)
    })
    
    repo_id_train = "TeslaInch/SCD-Instruction-Tuning"
    print(f"\nPushing training dataset to HuggingFace Hub: {repo_id_train}...")
    dataset_dict.push_to_hub(
        repo_id_train,
        private=False,
        commit_message="Updated SCD instruction-tuning dataset with Alpaca eval data."
    )
    
    # Also push Layer 6 to Eval Benchmark
    layer6_file = Path("data/eval/layer6_multiturn.json")
    if layer6_file.exists():
        print("\nLoading Layer 6 Multi-Turn dataset...")
        with open(layer6_file, "r", encoding="utf-8") as f:
            layer6_data = json.load(f)
            
        # Clean records into string format for PyArrow compatibility
        clean_layer6 = []
        for r in layer6_data:
            clean_layer6.append({k: str(v) if v is not None else "" for k, v in r.items()})
            
        ds_layer6 = Dataset.from_list(clean_layer6)
        repo_id_eval = "TeslaInch/SCD-Eval-Benchmark"
        print(f"Pushing layer6_multiturn to {repo_id_eval}...")
        ds_layer6.push_to_hub(repo_id_eval, config_name="layer6_multiturn", split="test", private=False)
    
    print("\nSuccess! Both datasets have been published.")

if __name__ == "__main__":
    main()
