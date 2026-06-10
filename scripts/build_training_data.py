import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
import re
import random

# Configuration
KEYWORDS = [
    r"sickle cell", r"scd", r"haematology", r"hematology", 
    r"hemoglobinopathy", r"vaso-occlusive", r"hydroxyurea", 
    r"haemoglobin s", r"hemoglobin s", r"hb s"
]
CUSTOM_QA_FILE = Path("data/train/custom_sickle_cell_qa.json")
OUTPUT_FILE = Path("data/train/instruction_dataset.json")

def contains_keywords(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    return any(re.search(kw, text) for kw in KEYWORDS)

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove multiple spaces, newlines, etc.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_training_data():
    print("Downloading medalpaca/medical_meadow_medqa dataset from HuggingFace...")
    # This dataset typically has columns: input, output, instruction
    dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")
    
    print(f"Total rows in medical_meadow_medqa: {len(dataset)}")
    
    filtered_examples = []
    seen_outputs = set()
    
    for row in dataset:
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output_text = row.get("output", "")
        
        # Combine text to search for keywords
        combined_text = f"{instruction} {input_text} {output_text}"
        
        if contains_keywords(combined_text):
            # Clean text
            instruction = clean_text(instruction)
            input_text = clean_text(input_text)
            output_text = clean_text(output_text)
            
            # Deduplicate by exact output (sometimes identical questions have identical answers)
            if output_text in seen_outputs:
                continue
            seen_outputs.add(output_text)
            
            filtered_examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "source": "medical_meadow_medqa"
            })
            
    print(f"Filtered {len(filtered_examples)} sickle cell/hematology examples from HuggingFace.")
    
    # Load custom hand-crafted Q&As if available
    custom_examples = []
    if CUSTOM_QA_FILE.exists():
        with open(CUSTOM_QA_FILE, "r", encoding="utf-8") as f:
            try:
                custom_data = json.load(f)
                for item in custom_data:
                    custom_examples.append({
                        "instruction": clean_text(item.get("instruction", "")),
                        "input": clean_text(item.get("input", "")),
                        "output": clean_text(item.get("output", "")),
                        "source": "custom_handcrafted"
                    })
                print(f"Loaded {len(custom_examples)} hand-crafted examples.")
            except json.JSONDecodeError:
                print(f"Warning: {CUSTOM_QA_FILE} is not valid JSON. Skipping.")
    else:
        print(f"Note: {CUSTOM_QA_FILE} not found. Proceeding without hand-crafted data.")
        
    # Load converted Alpaca eval layers
    alpaca_files = ["layer1_alpaca.json", "layer3_alpaca.json", "layer4_alpaca.json"]
    eval_examples = []
    for af in alpaca_files:
        path = Path(f"data/train/{af}")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                eval_examples.extend(data)
                print(f"Loaded {len(data)} examples from {af}")
                
    # Merge datasets
    merged_dataset = filtered_examples + custom_examples + eval_examples
    
    # Deduplicate on exact input match (to avoid duplicate questions)
    final_dataset = []
    seen_inputs = set()
    for item in merged_dataset:
        input_text = item.get("input", "").strip()
        if input_text in seen_inputs:
            continue
        seen_inputs.add(input_text)
        final_dataset.append(item)
    
    # Shuffle dataset
    random.seed(42)
    random.shuffle(final_dataset)
    
    # Split into Train / Validation (90/10)
    split_idx = int(len(final_dataset) * 0.9)
    train_split = final_dataset[:split_idx]
    val_split = final_dataset[split_idx:]
    
    # Save to disk as scd_training_final.json
    final_output_file = Path("data/train/scd_training_final.json")
    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "train": train_split,
        "validation": val_split
    }
    
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    # Print summary
    print("\n--- Final Dataset Summary ---")
    print(f"medical_meadow_medqa: {len(filtered_examples)}")
    print(f"custom_handcrafted: {len(custom_examples)}")
    print(f"converted_eval_layers: {len(eval_examples)}")
    print(f"Total before deduplication: {len(merged_dataset)}")
    print(f"Total after deduplication: {len(final_dataset)}")
    print(f"Train split size: {len(train_split)}")
    print(f"Validation split size: {len(val_split)}")
    print(f"\nSuccess! Final instruction dataset saved to {final_output_file}")

if __name__ == "__main__":
    build_training_data()
