import json
import os
import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push to HF after cleaning")
    args = parser.parse_args()

    input_file = Path("data/train/scd_training_final.json")
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords = ["sickle", "hbss", "hbsc", "hydroxyurea", "vaso-occlusive", "dactylitis", "hemoglobin s", "scd", "acute chest", "priapism", "sequestration"]

    def is_relevant(record):
        instruction = record.get("instruction", "").lower()
        input_text = record.get("input", "").lower()
        output_text = record.get("output", "").lower()
        
        # 1. If output has the keywords, it's relevant
        if any(kw in output_text for kw in keywords):
            return True
            
        # 2. If it is NOT a multiple choice question (e.g. from MedQA), check whole input and instruction
        if "medical_meadow_medqa" not in record.get("source", "") and not re.search(r"\?\s*\{", input_text):
            if any(kw in input_text for kw in keywords) or any(kw in instruction for kw in keywords):
                return True
            return False
            
        # 3. For multiple choice questions, split the question body from the options
        question_body = input_text
        match = re.search(r"(\?\?|\?)\s*\{", input_text)
        if match:
            idx = match.start()
            question_body = input_text[:idx]
            
        # If any keyword is in the question body, it's relevant
        if any(kw in question_body for kw in keywords):
            return True
            
        # Otherwise, the keyword is only in the distractors, so it's NOT relevant
        return False

    def clean_records(records):
        cleaned = []
        removed_not_relevant = 0
        removed_short_output = 0
        replaced_instruction = 0

        for r in records:
            output = r.get("output", "")
            if not output or len(output.strip()) < 20:
                removed_short_output += 1
                continue
            
            if not is_relevant(r):
                removed_not_relevant += 1
                continue

            instruction = r.get("instruction", "")
            if "please answer with one of the option" in instruction.lower():
                r["instruction"] = "You are a medical AI assistant specialised in sickle cell disease. Answer the following clinical question accurately and completely."
                replaced_instruction += 1
            
            cleaned.append(r)
            
        return cleaned, removed_not_relevant, removed_short_output, replaced_instruction

    if isinstance(data, dict):
        train_clean, rnr, rso, ri = clean_records(data.get("train", []))
        val_clean, rnr_v, rso_v, ri_v = clean_records(data.get("validation", []))
        
        data["train"] = train_clean
        if "validation" in data:
            data["validation"] = val_clean
            
        total_removed = rnr + rso + rnr_v + rso_v
        total_remain = len(train_clean) + len(val_clean)
        
        print("Summary of Cleaning:")
        print(f"- Removed (Not sickle cell related): {rnr + rnr_v}")
        print(f"- Removed (Output empty or < 20 chars): {rso + rso_v}")
        print(f"- Total Removed: {total_removed}")
        print(f"- Instructions Replaced: {ri + ri_v}")
        print(f"- Total Remaining Records: {total_remain}")
    else:
        cleaned, rnr, rso, ri = clean_records(data)
        data = cleaned
        print("Summary of Cleaning:")
        print(f"- Removed (Not sickle cell related): {rnr}")
        print(f"- Removed (Output empty or < 20 chars): {rso}")
        print(f"- Total Removed: {rnr + rso}")
        print(f"- Instructions Replaced: {ri}")
        print(f"- Total Remaining Records: {len(data)}")

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved cleaned dataset to {input_file}.")
    
    if args.push:
        print("Pushing to HuggingFace...")
        # Re-using the push logic
        from datasets import Dataset, DatasetDict
        from huggingface_hub import login
        from dotenv import load_dotenv
        
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Error: HF_TOKEN not found in .env")
            return
            
        login(token=hf_token)
        
        if isinstance(data, dict):
            train_records = data.get("train", [])
            val_records = data.get("validation", [])
            dataset_dict = DatasetDict({
                "train": Dataset.from_list(train_records),
                "validation": Dataset.from_list(val_records)
            })
        else:
            dataset_dict = DatasetDict({
                "train": Dataset.from_list(data)
            })
            
        repo_id_train = "TeslaInch/SCD-Instruction-Tuning"
        dataset_dict.push_to_hub(
            repo_id_train,
            private=False,
            commit_message="Cleaned training dataset: removed irrelevant & short examples, fixed instructions."
        )
        print("Push successful!")

if __name__ == "__main__":
    main()
