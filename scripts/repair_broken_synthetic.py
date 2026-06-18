"""
repair_broken_synthetic.py
──────────────────────────────────────────────────────────────────
Surgically repairs the three synthetic JSON files with unescaped
double-quotes inside string values, then converts all records to
Alpaca instruction/input/output format and saves in-place.

Usage:
    python scripts/repair_broken_synthetic.py
"""

import json
import re
import sys
from pathlib import Path

INSTRUCTION = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer the following clinical question accurately and completely."
)

BROKEN_FILES = [
    "more_training_data_2.json",
    "more_training_data_4.json",
    "more_training_data_5.json",
]

SYNTHETIC_DIR = Path("data/synthetic")


def extract_records_raw(text: str) -> list[dict]:
    """
    Extract records from raw text even when the JSON is broken due to
    unescaped double-quotes inside answer strings.

    Strategy:
      1. Strip outer brackets.
      2. Split on record boundaries  },  {
      3. For each record block, use a regex to extract field values,
         with DOTALL to handle multi-line answers.

    Args:
        text: Raw content of the JSON file.

    Returns:
        List of extracted dicts with keys: question, answer, category, source.
    """
    text = text.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]

    # Split on record-level delimiters
    parts = re.split(r"\},\s*\{", text)

    records = []
    for part in parts:
        part = part.strip().lstrip("{").rstrip("}").strip()

        fields = {}
        for field in ["question", "answer", "category", "source"]:
            # Match "field": "value" where value can be anything (including embedded quotes)
            # up until the next known field key or end of the record block.
            pattern = (
                r'"' + field + r'"\s*:\s*"(.*?)"'
                r"(?=\s*,\s*\"(?:question|answer|category|source)\"|$)"
            )
            m = re.search(pattern, part, re.DOTALL)
            if m:
                val = m.group(1)
                # Normalize escaped newlines to spaces for cleaner training data
                val = val.replace("\\n", " ").replace("\\t", " ")
                fields[field] = val.strip()

        if fields.get("question") and fields.get("answer"):
            records.append(fields)

    return records


def to_alpaca(record: dict) -> dict:
    """
    Convert a raw extracted record to Alpaca format.

    Args:
        record: Dict with keys question, answer, source, category.

    Returns:
        Alpaca-format dict with instruction, input, output, source, category.
    """
    return {
        "instruction": INSTRUCTION,
        "input": record.get("question", "").strip(),
        "output": record.get("answer", "").strip(),
        "source": record.get("source", "synthetic"),
        "category": record.get("category", ""),
    }


def main() -> None:
    """
    Entry point: repair and convert each broken file, then verify all 6 files.
    """
    # ── Step 1: Repair the broken files ──────────────────────────────────────
    for fname in BROKEN_FILES:
        fpath = SYNTHETIC_DIR / fname
        if not fpath.exists():
            print(f"[SKIP] {fname} not found.")
            continue

        print(f"\n[Repairing] {fname}")
        raw = fpath.read_text(encoding="utf-8")
        records = extract_records_raw(raw)
        print(f"  Extracted {len(records)} records from raw text.")

        if not records:
            print(f"  [ERROR] Could not extract any records — skipping.")
            continue

        converted = [to_alpaca(r) for r in records if r.get("answer", "")]
        print(f"  Converted {len(converted)} records to Alpaca format.")

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=4, ensure_ascii=False)
        print(f"  Saved -> {fpath}")

    # ── Step 2: Final validation of all 6 files ───────────────────────────────
    print(f"\n{'=' * 55}")
    print("  FINAL VALIDATION")
    print(f"{'=' * 55}")

    all_ok = True
    total = 0
    for fpath in sorted(SYNTHETIC_DIR.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = list(data[0].keys()) if data else []
            has_alpaca = "input" in keys and "output" in keys and "instruction" in keys
            status = "OK " if has_alpaca else "SCHEMA MISMATCH"
            print(f"  {status:<6} {fpath.name:<35} {len(data):>4} records  keys={keys}")
            total += len(data)
        except Exception as e:
            print(f"  ERROR  {fpath.name:<35} {e}")
            all_ok = False

    print(f"{'=' * 55}")
    print(f"  Total records across all files: {total}")
    print(f"{'=' * 55}\n")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
