"""
build_v3_dataset.py
─────────────────────────────────────────────────────────────────────────────
Merges existing scd_training_v2.json with 118 synthetic records from
data/synthetic/, deduplicates, re-splits 90/10, saves as scd_training_v3.json,
and pushes to TeslaInch/SCD-Instruction-Tuning on HuggingFace.

Usage:
    python scripts/build_v3_dataset.py

Environment:
    HF_TOKEN must be set in .env
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# ── paths ─────────────────────────────────────────────────────────────────────

TRAIN_DIR = Path("data/train")
SYNTHETIC_DIR = Path("data/synthetic")

V2_FILE = TRAIN_DIR / "scd_training_v2.json"
V3_FILE = TRAIN_DIR / "scd_training_v3.json"

SYNTHETIC_FILES = [
    "more_training_data.json",
    "more_training_data_2.json",
    "more_training_data_3.json",
    "more_training_data_4.json",
    "more_training_data_5.json",
    "more_training_data_6.json",
]


# ── loading ───────────────────────────────────────────────────────────────────

def load_v2(filepath: Path) -> tuple[list[dict], list[dict]]:
    """
    Load train and validation splits from scd_training_v2.json.

    Args:
        filepath: Path to the v2 JSON file.

    Returns:
        Tuple of (train_records, val_records).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the expected keys are missing.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"v2 file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "train" not in data or "validation" not in data:
        raise ValueError(f"Expected 'train' and 'validation' keys in {filepath}")

    return data["train"], data["validation"]


def load_synthetic(synthetic_dir: Path, filenames: list[str]) -> list[dict]:
    """
    Load all synthetic JSON files and return a combined list of records.

    Args:
        synthetic_dir: Directory containing synthetic files.
        filenames:     List of filenames to load.

    Returns:
        Combined list of all synthetic records.

    Raises:
        FileNotFoundError: If any expected file is missing.
        ValueError: If a file is not a JSON list.
    """
    all_records = []
    for fname in filenames:
        fpath = synthetic_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Synthetic file not found: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{fname}: expected a JSON list, got {type(data).__name__}")

        all_records.extend(data)
        print(f"  Loaded {len(data):>4} records from {fname}")

    return all_records


# ── deduplication ─────────────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> list[dict]:
    """
    Remove duplicate records based on exact match of the 'input' field.
    Preserves order of first occurrence.

    Args:
        records: Full combined list of records.

    Returns:
        Deduplicated list.
    """
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
    """
    Shuffle and split records into train and validation sets.

    Args:
        records:      Full deduplicated list.
        val_ratio:    Fraction to use for validation (default 0.1).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_records, val_records).
    """
    rng = random.Random(random_state)
    shuffled = records[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(
    before_dedup: int,
    all_records: list[dict],
    train: list[dict],
    val: list[dict],
) -> None:
    """
    Print a detailed breakdown of the merged dataset by source.

    Args:
        before_dedup: Total count before deduplication.
        all_records:  Full merged, deduplicated list.
        train:        Training split.
        val:          Validation split.
    """
    source_counts = Counter(r.get("source", "unknown") for r in all_records)

    print(f"\n{'=' * 65}")
    print("  V3 DATASET SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Source':<44} {'Count':>5}")
    print(f"  {'-' * 51}")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source:<44} {count:>5}")
    print(f"  {'-' * 51}")
    print(f"  {'Total before dedup':<44} {before_dedup:>5}")
    print(f"  {'Total after dedup':<44} {len(all_records):>5}")
    print(f"  {'Duplicates removed':<44} {before_dedup - len(all_records):>5}")
    print(f"  {'-' * 51}")
    print(f"  {'Train split (90%)':<44} {len(train):>5}")
    print(f"  {'Validation split (10%)':<44} {len(val):>5}")
    print(f"{'=' * 65}\n")


# ── huggingface push ──────────────────────────────────────────────────────────

def push_to_huggingface(train: list[dict], val: list[dict], hf_token: str) -> None:
    """
    Push the v3 train/validation splits to TeslaInch/SCD-Instruction-Tuning.

    Args:
        train:    Training records.
        val:      Validation records.
        hf_token: HuggingFace API token.

    Raises:
        RuntimeError: If the push fails.
    """
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
            "Dataset v3: +118 synthetic PDF-extracted examples, "
            f"total {len(train) + len(val)} records."
        ),
    )

    print(f"\nSuccessfully pushed to {repo_id}")
    print(f"  Train records      : {len(train)}")
    print(f"  Validation records : {len(val)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: execute all 8 steps.
    """
    load_dotenv()

    # ── Step 1: Load v2 ───────────────────────────────────────────────────────
    print("\n[Step 1] Loading scd_training_v2.json...")
    try:
        train_v2, val_v2 = load_v2(V2_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    existing = train_v2 + val_v2
    print(f"  Loaded {len(train_v2)} train + {len(val_v2)} val = {len(existing)} total from v2.")

    # ── Step 2: Load synthetic ────────────────────────────────────────────────
    print("\n[Step 2] Loading synthetic files...")
    try:
        synthetic = load_synthetic(SYNTHETIC_DIR, SYNTHETIC_FILES)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Total synthetic records loaded: {len(synthetic)}")

    # ── Step 3: Combine ───────────────────────────────────────────────────────
    print("\n[Step 3] Combining existing train split + synthetic records...")
    combined = existing + synthetic
    before_dedup = len(combined)
    print(f"  Combined total: {before_dedup} records")

    # ── Step 4: Deduplicate ───────────────────────────────────────────────────
    print("\n[Step 4] Deduplicating on exact input field...")
    combined = deduplicate(combined)
    print(f"  After dedup: {len(combined)} unique records ({before_dedup - len(combined)} removed)")

    # ── Step 5: Re-split ──────────────────────────────────────────────────────
    print("\n[Step 5] Re-splitting 90/10 train/validation (random_state=42)...")
    train_v3, val_v3 = split_dataset(combined, val_ratio=0.1, random_state=42)
    print(f"  Train: {len(train_v3)}  |  Validation: {len(val_v3)}")

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    print(f"\n[Step 6] Saving to {V3_FILE}...")
    with open(V3_FILE, "w", encoding="utf-8") as f:
        json.dump({"train": train_v3, "validation": val_v3}, f, indent=2, ensure_ascii=False)
    print(f"  Saved successfully.")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    print_summary(before_dedup, combined, train_v3, val_v3)

    # ── Step 8: Push to HuggingFace ───────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN not found in .env — skipping push.", file=sys.stderr)
        sys.exit(1)

    print("[Step 8] Pushing v3 to HuggingFace...")
    try:
        push_to_huggingface(train_v3, val_v3, hf_token)
    except Exception as e:
        print(f"  [ERROR] Push failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
