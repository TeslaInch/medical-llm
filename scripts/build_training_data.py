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
        
    # Merge datasets
    final_dataset = filtered_examples + custom_examples
    
    # Shuffle dataset
    random.seed(42)
    random.shuffle(final_dataset)
    
    # Save to disk
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Success! Final instruction dataset saved to {OUTPUT_FILE} ({len(final_dataset)} total examples)")

if __name__ == "__main__":
    build_training_data()
