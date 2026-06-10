"""
push_eval_to_hf.py
──────────────────
Pushes the 5-layer SCD Evaluation Benchmark to the HuggingFace Hub.
Each layer is stored as a separate split or configuration in the dataset.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from huggingface_hub import login

def load_layer(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "fullContent" in data:
        return data["fullContent"]
    return data

def main():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in .env")
        return

    login(token=hf_token)
    
    # Define paths
    eval_dir = Path("data/eval")
    layer_files = {
        "layer1_custom_notes": eval_dir / "layer1_custom_notes.json",
        "layer2_mcq_benchmark": eval_dir / "layer2_benchmark.json",
        "layer3_combined_reasoning": eval_dir / "layer3_combined.json",
        "layer4_clinical_cases": eval_dir / "layer4_clinical_cases.json",
        "layer5_ash_guidelines": eval_dir / "layer5_ASH.json"
    }

    print("Loading layer data...")
    dataset_dict = {}
    
    # Dump standardized JSON to temp files to avoid nested array schema issues
    temp_dir = Path("data/eval/hf_temp")
    temp_dir.mkdir(exist_ok=True)
    
    for layer_name, path in layer_files.items():
        if not path.exists():
            continue
            
        records = load_layer(path)
        # Force all values to string to prevent PyArrow type conflicts
        clean_records = []
        for r in records:
            clean_records.append({k: str(v) if v is not None else "" for k, v in r.items()})
            
        dataset_dict[layer_name] = Dataset.from_list(clean_records)

    repo_id = "TeslaInch/SCD-Eval-Benchmark"
    print(f"\\nPushing dataset to HuggingFace Hub: {repo_id}...")
    
    for layer_name, ds in dataset_dict.items():
        print(f"Pushing config: {layer_name}...")
        ds.push_to_hub(repo_id, config_name=layer_name, split="test", private=False)
        
    print("\\nSuccess! The benchmark has been published.")
    print(f"View it here: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
