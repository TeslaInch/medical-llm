"""
convert_eval_to_training.py
───────────────────────────
Converts high-quality evaluation data (Layers 1, 3, 4) into Alpaca instruction-tuning format.
Filters out short answers and removes duplicates based on exact question match.
"""

import json
from pathlib import Path

EVAL_DIR = Path("data/eval")
TRAIN_DIR = Path("data/train")

def load_json(filepath: Path) -> list:
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "fullContent" in data:
        return data["fullContent"]
    return data

def save_json(filepath: Path, data: list):
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def is_valid_answer(answer: str) -> bool:
    if not answer or not isinstance(answer, str):
        return False
    return len(answer.strip()) >= 20

def process_layer(input_filename: str, output_filename: str, seen_questions: set, is_layer4: bool = False):
    input_path = EVAL_DIR / input_filename
    output_path = TRAIN_DIR / output_filename
    
    records = load_json(input_path)
    if not records:
        return
        
    converted = []
    skipped = 0
    duplicates = 0
    
    instruction = "You are a medical AI assistant specialised in sickle cell disease. Answer the following clinical question accurately and completely."
    
    for r in records:
        question = r.get("question", "").strip()
        answer = r.get("answer", r.get("gold_answer", "")).strip()
        
        if not is_valid_answer(answer):
            skipped += 1
            continue
            
        if question in seen_questions:
            duplicates += 1
            continue
            
        seen_questions.add(question)
        
        if is_layer4:
            case = r.get("case", "").strip()
            input_field = f"Clinical case:\n{case}\n\nQuestion: {question}"
        else:
            input_field = question
            
        alpaca_entry = {
            "instruction": instruction,
            "input": input_field,
            "output": answer,
        }
        
        if "source" in r:
            alpaca_entry["source"] = r["source"]
        if "category" in r:
            alpaca_entry["category"] = r["category"]
            
        converted.append(alpaca_entry)
        
    save_json(output_path, converted)
    print(f"{input_filename}: Converted {len(converted)}, Skipped {skipped}, Duplicates removed {duplicates}")

def main():
    seen_questions = set()
    print("Starting conversion to Alpaca format...\n")
    
    # Layer 1
    process_layer("layer1_custom_notes.json", "layer1_alpaca.json", seen_questions)
    
    # Layer 3
    process_layer("layer3_combined.json", "layer3_alpaca.json", seen_questions)
    
    # Layer 4 (Requires special case prepend)
    process_layer("layer4_clinical_cases.json", "layer4_alpaca.json", seen_questions, is_layer4=True)
    
    print("\nConversion complete. Files saved to data/train/")

if __name__ == "__main__":
    main()
