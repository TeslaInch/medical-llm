"""
expand_training_data.py
────────────────────────────────────────────────────────────────────────────
Expands the SCD fine-tuning dataset from ~140 to 400+ examples by:
  1. Filtering layer2_benchmark.json for SCD-relevant questions → Alpaca format
  2. Converting layer5_ASH.json → Alpaca format
  3. Converting layer6_multiturn.json (multi-turn conversations) → Alpaca format
  4. Merging all existing + new training files, deduplicating, splitting 90/10
  5. Saving as data/train/scd_training_v2.json
  6. Pushing to HuggingFace as TeslaInch/SCD-Instruction-Tuning

Usage:
    python scripts/expand_training_data.py

Environment:
    HF_TOKEN must be set in .env
"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── constants ─────────────────────────────────────────────────────────────────

SCD_KEYWORDS = [
    "sickle",
    "hbss",
    "hbsc",
    "hydroxyurea",
    "vaso-occlusive",
    "dactylitis",
    "hemoglobin s",
    "haemoglobin s",
    "scd",
    "acute chest",
    "priapism",
    "sequestration",
    "fetal hemoglobin",
    "hbf",
    "transcranial doppler",
    "aplastic crisis",
    "parvovirus",
    "exchange transfusion",
    "penicillin prophylaxis",
    "sickle cell",
]

INSTRUCTION = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer the following clinical question accurately and completely."
)

EVAL_DIR = Path("data/eval")
TRAIN_DIR = Path("data/train")

EXISTING_TRAIN_FILES = [
    TRAIN_DIR / "layer1_alpaca.json",
    TRAIN_DIR / "layer3_alpaca.json",
    TRAIN_DIR / "layer4_alpaca.json",
    TRAIN_DIR / "instruction_dataset.json",
]

NEW_FILES = {
    "layer2": TRAIN_DIR / "layer2_scd_filtered_alpaca.json",
    "layer5": TRAIN_DIR / "layer5_alpaca.json",
    "layer6": TRAIN_DIR / "layer6_alpaca.json",
}

OUTPUT_FILE = TRAIN_DIR / "scd_training_v2.json"


# ── helper: SCD relevance check ───────────────────────────────────────────────

def is_scd_relevant(text: str) -> bool:
    """
    Return True if any SCD keyword appears in the given text (case-insensitive).

    Args:
        text: The string to check.

    Returns:
        True if at least one SCD keyword is found, False otherwise.
    """
    lower = text.lower()
    return any(kw in lower for kw in SCD_KEYWORDS)


# ── step 1: layer2 filter → alpaca ────────────────────────────────────────────

def build_layer2_alpaca() -> list[dict]:
    """
    Load layer2_benchmark.json, filter for SCD-relevant records using
    keyword matching on question + answer, and convert to Alpaca format.

    Returns:
        List of Alpaca-format dicts.

    Raises:
        FileNotFoundError: If layer2_benchmark.json is missing.
        ValueError: If the file structure is unexpected.
    """
    filepath = EVAL_DIR / "layer2_benchmark.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tolerate wrapper dict
    if isinstance(data, dict) and "fullContent" in data:
        records = data["fullContent"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected structure in {filepath}")

    results = []
    for record in records:
        question = record.get("question", "")
        answer = record.get("answer", "")

        # Skip if output is too short to be useful
        if not answer or len(answer.strip()) < 20:
            continue

        # Only keep SCD-relevant records based on question OR answer
        combined = question + " " + answer
        if not is_scd_relevant(combined):
            continue

        results.append({
            "instruction": INSTRUCTION,
            "input": question.strip(),
            "output": answer.strip(),
            "source": "layer2_mcq_filtered",
            "category": "multiple_choice",
        })

    return results


# ── step 2: layer5 → alpaca ───────────────────────────────────────────────────

def build_layer5_alpaca() -> list[dict]:
    """
    Load layer5_ASH.json and convert every record to Alpaca format.

    Returns:
        List of Alpaca-format dicts.

    Raises:
        FileNotFoundError: If layer5_ASH.json is missing.
    """
    filepath = EVAL_DIR / "layer5_ASH.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = []
    for record in records:
        question = record.get("question", "").strip()
        answer = record.get("answer", "").strip()

        if not question or not answer:
            continue

        # Append rationale to the answer for richer training signal
        rationale = record.get("rationale", "").strip()
        full_answer = answer
        if rationale:
            full_answer = f"{answer}\n\nRationale: {rationale}"

        results.append({
            "instruction": INSTRUCTION,
            "input": question,
            "output": full_answer,
            "source": "layer5_ASH_guidelines",
            "category": record.get("category", ""),
        })

    return results


# ── step 3: layer6 multi-turn → alpaca ───────────────────────────────────────

def build_layer6_alpaca() -> list[dict]:
    """
    Load layer6_multiturn.json and flatten each conversation into standalone
    Alpaca training examples. For turns 2+, the previous turn's question and
    answer are prepended to the input as context so the model learns from
    conversational flow.

    Returns:
        List of Alpaca-format dicts.

    Raises:
        FileNotFoundError: If layer6_multiturn.json is missing.
    """
    filepath = EVAL_DIR / "layer6_multiturn.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    results = []
    for convo in conversations:
        convo_id = convo.get("id", "unknown")
        category = convo.get("category", "multi_turn_clinical")
        turns = convo.get("conversation", [])

        prev_user = ""
        prev_answer = ""

        for turn_data in turns:
            turn_num = turn_data.get("turn", 0)
            user_msg = turn_data.get("user", "").strip()
            gold_resp = turn_data.get("gold_response", "").strip()

            if not user_msg or not gold_resp:
                continue

            # For turn 1, the input is just the question
            if turn_num == 1 or not prev_user:
                input_text = user_msg
            else:
                # For subsequent turns, prepend the previous Q&A as context
                input_text = (
                    f"[Previous Question]\n{prev_user}\n\n"
                    f"[Previous Answer]\n{prev_answer}\n\n"
                    f"[Follow-up Question]\n{user_msg}"
                )

            results.append({
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": gold_resp,
                "source": "layer6_multiturn",
                "category": category,
            })

            # Carry forward for next turn
            prev_user = user_msg
            prev_answer = gold_resp

    return results


# ── step 4: load existing training files ──────────────────────────────────────

def load_existing_train_files() -> list[dict]:
    """
    Load all existing Alpaca-format training files from data/train/.

    Returns:
        Combined list of all training records from existing layer files.
    """
    all_records = []
    for fpath in EXISTING_TRAIN_FILES:
        if not fpath.exists():
            print(f"  [!] Skipping missing file: {fpath.name}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_records.extend(data)
        else:
            print(f"  [!] Unexpected structure in {fpath.name}, skipping.")
    return all_records


# ── step 4b: deduplicate ──────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> list[dict]:
    """
    Remove duplicate records based on exact match of the `input` field.

    Args:
        records: Full list of training records.

    Returns:
        Deduplicated list, preserving order of first occurrence.
    """
    seen = set()
    unique = []
    for r in records:
        key = r.get("input", "").strip()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ── step 4c: train/val split ──────────────────────────────────────────────────

def split_dataset(records: list[dict], val_ratio: float = 0.1, random_state: int = 42) -> tuple[list, list]:
    """
    Split records into train and validation sets.

    Args:
        records:      Full list of deduplicated records.
        val_ratio:    Fraction to use as validation (default 0.1 = 10%).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_records, val_records).
    """
    rng = random.Random(random_state)
    shuffled = records[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


# ── step 4d: summary ──────────────────────────────────────────────────────────

def print_summary(all_records: list[dict], train: list[dict], val: list[dict]) -> None:
    """
    Print a detailed breakdown of the merged dataset by source.

    Args:
        all_records: Full merged, deduplicated list before splitting.
        train:       Training split records.
        val:         Validation split records.
    """
    from collections import Counter
    source_counts = Counter(r.get("source", "unknown") for r in all_records)

    print(f"\n{'=' * 65}")
    print("  TRAINING DATASET EXPANSION SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Source':<40} {'Count':>6}")
    print(f"  {'-' * 48}")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source:<40} {count:>6}")
    print(f"  {'-' * 48}")
    print(f"  {'TOTAL (deduplicated)':<40} {len(all_records):>6}")
    print(f"  {'Train split (90%)':<40} {len(train):>6}")
    print(f"  {'Validation split (10%)':<40} {len(val):>6}")
    print(f"{'=' * 65}\n")


# ── step 5: push to HuggingFace ───────────────────────────────────────────────

def push_to_huggingface(train: list[dict], val: list[dict], hf_token: str) -> None:
    """
    Push the train/validation splits to TeslaInch/SCD-Instruction-Tuning on HuggingFace.

    Args:
        train:     List of training records.
        val:       List of validation records.
        hf_token:  HuggingFace API token.

    Raises:
        RuntimeError: If push fails.
    """
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login

    print("Logging into HuggingFace...")
    login(token=hf_token)

    # Normalize keys so both splits have identical columns
    all_keys = set()
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
        commit_message="Expanded SCD dataset v2: added layer2 filtered, layer5 ASH, layer6 multi-turn.",
    )
    print(f"\n✓ Successfully pushed to {repo_id}")
    print(f"  Train records : {len(train)}")
    print(f"  Val records   : {len(val)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: run all five steps in order.
    """
    load_dotenv()

    # ── Step 1: Layer 2 filter ────────────────────────────────────────────────
    print("\n[Step 1] Filtering layer2_benchmark.json for SCD relevance...")
    try:
        layer2_records = build_layer2_alpaca()
    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {len(layer2_records)} SCD-relevant records extracted from layer2.")

    with open(NEW_FILES["layer2"], "w", encoding="utf-8") as f:
        json.dump(layer2_records, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {NEW_FILES['layer2']}")

    # ── Step 2: Layer 5 convert ───────────────────────────────────────────────
    print("\n[Step 2] Converting layer5_ASH.json to Alpaca format...")
    try:
        layer5_records = build_layer5_alpaca()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {len(layer5_records)} records converted from layer5.")

    with open(NEW_FILES["layer5"], "w", encoding="utf-8") as f:
        json.dump(layer5_records, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {NEW_FILES['layer5']}")

    # ── Step 3: Layer 6 multi-turn convert ───────────────────────────────────
    print("\n[Step 3] Converting layer6_multiturn.json to Alpaca format...")
    try:
        layer6_records = build_layer6_alpaca()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {len(layer6_records)} turn examples extracted from layer6.")

    with open(NEW_FILES["layer6"], "w", encoding="utf-8") as f:
        json.dump(layer6_records, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {NEW_FILES['layer6']}")

    # ── Step 4: Merge, deduplicate, split ─────────────────────────────────────
    print("\n[Step 4] Merging all training files...")
    existing = load_existing_train_files()
    print(f"  → Loaded {len(existing)} records from existing train files.")

    all_records = existing + layer2_records + layer5_records + layer6_records
    print(f"  → Combined total before dedup: {len(all_records)}")

    all_records = deduplicate(all_records)
    print(f"  → After deduplication: {len(all_records)} unique records.")

    train_records, val_records = split_dataset(all_records, val_ratio=0.1, random_state=42)

    output = {"train": train_records, "validation": val_records}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  → Saved merged dataset to {OUTPUT_FILE}")

    print_summary(all_records, train_records, val_records)

    # ── Step 5: Push to HuggingFace ──────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN not found in .env — skipping HuggingFace push.", file=sys.stderr)
        sys.exit(1)

    print("[Step 5] Pushing to HuggingFace...")
    try:
        push_to_huggingface(train_records, val_records, hf_token)
    except Exception as e:
        print(f"  [ERROR] HuggingFace push failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
