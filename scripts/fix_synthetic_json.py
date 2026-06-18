"""
fix_synthetic_json.py
─────────────────────────────────────────────────────────────────────────────
Validates and fixes all JSON files in data/synthetic/:
  1. Repairs broken JSON (unescaped quotes inside string values)
  2. Converts question/answer schema → Alpaca instruction/input/output schema
  3. Removes records with empty input or output
  4. Saves fixed files back in-place

Usage:
    python scripts/fix_synthetic_json.py
"""

import json
import re
import sys
from pathlib import Path

INSTRUCTION = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer the following clinical question accurately and completely."
)

SYNTHETIC_DIR = Path("data/synthetic")


def repair_json(raw: str, filepath: Path) -> list:
    """
    Attempt to parse JSON, and if it fails, repair common issues:
    - Unescaped double-quote characters embedded inside string values
      (e.g. the Abiku/Ogbanje line split by an accidental closing quote)

    Args:
        raw:      Raw text content of the JSON file.
        filepath: Used for error messages only.

    Returns:
        Parsed list of records.

    Raises:
        ValueError: If the file cannot be repaired.
    """
    # First attempt: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [!] JSON error in {filepath.name} at line {e.lineno}, col {e.colno}: {e.msg}")

    # Repair strategy: the most common pattern in these files is an answer string
    # that has an unescaped double-quote, causing the JSON parser to think the
    # string ended early. For example:
    #   "answer": "...such as \"Abiku\" or \"Ogbanje,\n         \" which attribute..."
    # The trailing quote+newline+spaces+quote creates two separate broken tokens.
    #
    # We fix this by using a regex that finds the exact broken pattern:
    # a closing quote+comma+newline+spaces+opening quote INSIDE what should be
    # a single JSON string value. We merge these back into a single escaped string.

    print(f"  [~] Attempting regex repair on {filepath.name}...")

    # Strategy: find all string literals in the JSON and re-escape inner quotes.
    # We do this by scanning character by character for a more robust fix.
    fixed = _manual_repair(raw)

    try:
        result = json.loads(fixed)
        print(f"  [+] Successfully repaired {filepath.name}")
        return result
    except json.JSONDecodeError as e2:
        raise ValueError(
            f"Could not repair {filepath.name}: {e2.msg} at line {e2.lineno}, col {e2.colno}"
        )


def _manual_repair(raw: str) -> str:
    """
    Scan JSON text and fix broken string values where a double-quote
    accidentally terminates a string early, with the remainder appearing
    as a separate orphan token on the next line.

    The specific pattern we target:
        ",\n         " which...   ->  , which...
    where the comma+newline+spaces+quote is inside a string value.

    Args:
        raw: Raw JSON text.

    Returns:
        Repaired JSON text.
    """
    # Pattern: a double-quote followed by a comma+newline+whitespace+double-quote
    # that appears to be an incorrectly split string continuation.
    # We replace:   ,"  (with optional whitespace)
    # where the surrounding context is inside a string value.
    # Simple heuristic: find lines where a JSON value appears to continue
    # with a bare string that is not a key.

    lines = raw.split("\n")
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect lines that end with a stray pattern like: ",
        # followed by a next line starting with whitespace + "
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # If current line ends with a quote+comma (broken string end)
            # and next line is just whitespace + a string continuation
            stripped_next = next_line.strip()
            if (line.rstrip().endswith('",') and
                    stripped_next.startswith('"') and
                    not stripped_next.startswith('"category"') and
                    not stripped_next.startswith('"source"') and
                    not stripped_next.startswith('"question"') and
                    not stripped_next.startswith('"answer"') and
                    not stripped_next.startswith('"instruction"') and
                    not stripped_next.startswith('"input"') and
                    not stripped_next.startswith('"output"') and
                    not stripped_next == '",' and
                    "{" not in stripped_next and
                    "}" not in stripped_next):
                # Merge: remove the closing quote from end of current line
                # and the opening quote from start of next line
                current_fixed = line.rstrip()[:-2]  # remove the trailing ",
                next_fixed = stripped_next[1:]        # remove the leading "
                merged = current_fixed + ", " + next_fixed
                result_lines.append(merged)
                i += 2
                continue
        result_lines.append(line)
        i += 1

    return "\n".join(result_lines)


def convert_to_alpaca(records: list, source_file: str) -> list:
    """
    Convert records from question/answer schema to Alpaca instruction/input/output format.
    If records are already in Alpaca format, they are returned unchanged.

    Args:
        records:     List of dicts from the JSON file.
        source_file: Filename for logging purposes.

    Returns:
        List of Alpaca-formatted dicts.
    """
    if not records:
        return []

    # Check if already Alpaca format
    if "input" in records[0] and "output" in records[0]:
        print(f"  [=] Already in Alpaca format — validating only.")
        return _validate_alpaca(records, source_file)

    converted = []
    skipped = 0
    for i, record in enumerate(records):
        question = record.get("question", "").strip()
        answer = record.get("answer", "").strip()

        if not question:
            print(f"  [!] Record {i} has empty question — skipping.")
            skipped += 1
            continue
        if not answer or len(answer) < 20:
            print(f"  [!] Record {i} has empty/short answer — skipping.")
            skipped += 1
            continue

        converted.append({
            "instruction": INSTRUCTION,
            "input": question,
            "output": answer,
            "source": record.get("source", "synthetic"),
            "category": record.get("category", ""),
        })

    if skipped:
        print(f"  [!] Skipped {skipped} records with empty/short fields.")

    return converted


def _validate_alpaca(records: list, source_file: str) -> list:
    """
    Validate Alpaca-format records and remove any with empty input or output.

    Args:
        records:     List of Alpaca-format dicts.
        source_file: Filename for logging purposes.

    Returns:
        Cleaned list of valid records.
    """
    valid = []
    for i, r in enumerate(records):
        if not r.get("input", "").strip():
            print(f"  [!] Record {i} empty input — skipping.")
            continue
        if not r.get("output", "").strip() or len(r.get("output", "")) < 20:
            print(f"  [!] Record {i} empty/short output — skipping.")
            continue
        valid.append(r)
    return valid


def process_file(fpath: Path) -> bool:
    """
    Process a single synthetic JSON file: repair, convert, validate, and save.

    Args:
        fpath: Path to the JSON file.

    Returns:
        True if successful, False if an unrecoverable error occurred.
    """
    print(f"\n--- {fpath.name} ---")
    raw = fpath.read_text(encoding="utf-8")

    try:
        records = repair_json(raw, fpath)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return False

    if not isinstance(records, list):
        print(f"  [ERROR] Top-level is not a list — skipping.")
        return False

    converted = convert_to_alpaca(records, fpath.name)

    if not converted:
        print(f"  [ERROR] No valid records after conversion — skipping save.")
        return False

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    print(f"  [OK] {len(converted)} records saved in Alpaca format.")
    return True


def main() -> None:
    """
    Entry point: process all JSON files in data/synthetic/.
    """
    files = sorted(SYNTHETIC_DIR.glob("*.json"))
    if not files:
        print(f"[ERROR] No JSON files found in {SYNTHETIC_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing {len(files)} files in {SYNTHETIC_DIR.resolve()}\n")

    success = 0
    failed = 0
    for fpath in files:
        ok = process_file(fpath)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Done: {success} fixed, {failed} failed.")
    print(f"{'=' * 50}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
