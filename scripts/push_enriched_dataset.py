"""
push_enriched_dataset.py
─────────────────────────────────────────────────────────────────────────────
Loads the full_enriched_dataset.jsonl data, deduplicates it, 
splits it 90/10 (train/validation), saves it locally as 
scd_training_v5.json, and pushes it to TeslaInch/SCD-Instruction-Tuning.

Usage:
    python scripts/push_enriched_dataset.py

Environment:
    HF_TOKEN must be set in .env
"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── paths ─────────────────────────────────────────────────────────────────────

TRAIN_DIR = Path("data/train")
NEW_FILE = TRAIN_DIR / "full_enriched_dataset.jsonl"
V5_FILE = TRAIN_DIR / "scd_training_v5.json"

# ── loading ───────────────────────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line {i+1} due to JSON error: {e}")
                
    return records

# ── deduplication ─────────────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for r in records:
        key = r.get("input", "").strip()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# ── splitting ─────────────────────────────────────────────────────────────────

def split_dataset(
    records: list[dict],
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(random_state)
    shuffled = records[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]

# ── huggingface push ──────────────────────────────────────────────────────────

def push_to_huggingface(train: list[dict], val: list[dict], hf_token: str) -> None:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login

    print("Logging into HuggingFace...")
    login(token=hf_token)

    # Normalize keys so both splits have identical columns
    all_keys: set[str] = set()
    for r in train + val:
        all_keys.update(r.keys())
    for r in train + val:
        for k in all_keys:
            if k not in r:
                r[k] = ""

    dataset_dict = DatasetDict({
        "train":      Dataset.from_list(train),
        "validation": Dataset.from_list(val),
    })

    repo_id = "TeslaInch/SCD-Instruction-Tuning"
    print(f"Pushing to {repo_id}...")
    dataset_dict.push_to_hub(
        repo_id,
        private=False,
        commit_message=(
            f"Dataset updated with fully enriched {len(train) + len(val)} records."
        ),
    )

    print(f"\nSuccessfully pushed to {repo_id}")
    print(f"  Train records      : {len(train)}")
    print(f"  Validation records : {len(val)}")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    print(f"\n[Step 1] Loading {NEW_FILE.name}...")
    try:
        records = load_jsonl(NEW_FILE)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"  Loaded {len(records)} total records.")

    print("\n[Step 2] Deduplicating on exact input field...")
    unique_records = deduplicate(records)
    print(f"  After dedup: {len(unique_records)} unique records ({len(records) - len(unique_records)} removed)")

    print("\n[Step 3] Splitting 90/10 train/validation (random_state=42)...")
    train_split, val_split = split_dataset(unique_records, val_ratio=0.1, random_state=42)
    print(f"  Train: {len(train_split)}  |  Validation: {len(val_split)}")

    print(f"\n[Step 4] Saving locally to {V5_FILE}...")
    with open(V5_FILE, "w", encoding="utf-8") as f:
        json.dump({"train": train_split, "validation": val_split}, f, indent=2, ensure_ascii=False)
    print(f"  Saved successfully.")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN not found in .env — skipping push.", file=sys.stderr)
        sys.exit(1)

    print("\n[Step 5] Pushing to HuggingFace...")
    try:
        push_to_huggingface(train_split, val_split, hf_token)
    except Exception as e:
        print(f"  [ERROR] Push failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
