"""
extract_questions.py
────────────────────
Extracts the `question` field from every eval JSON file under data/eval/
and writes all questions to a single consolidated JSON file.

Special handling:
  - Layer 4 (clinical cases): the `case` field (patient scenario) is
    prepended to the `question` text so that the question retains its
    full clinical context.
  - The baselines/ subfolder is explicitly excluded — it contains model
    evaluation results, not source questions.

Usage:
    python scripts/extract_questions.py
    python scripts/extract_questions.py --input-dir data/eval --output scripts/all_questions.json

Output schema (one object per question):
    {
        "index":    <int>       — 1-based position in the final list,
        "id":       <str>       — original record id,
        "layer":    <int>       — source layer number,
        "source":   <str>       — source document name,
        "category": <str>       — clinical category tag,
        "question": <str>       — full question text (case prepended for Layer 4)
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── file map ─────────────────────────────────────────────────────────────────
EVAL_FILES = {
    1: "layer1_custom_notes.json",
    2: "layer2_benchmark.json",
    3: "layer3_combined.json",
    4: "layer4_clinical_cases.json",
    5: "layer5_ASH.json",
}

# Subfolder to exclude — contains model outputs, not source questions
EXCLUDED_SUBDIRS = {"baselines"}


# ── discovery ────────────────────────────────────────────────────────────────
def discover_json_files(input_dir: Path) -> dict[int, Path]:
    """
    Resolve the four known eval JSON file paths from the input directory.

    Returns a dict mapping layer number → resolved Path.
    Raises FileNotFoundError if any expected file is missing.

    Args:
        input_dir: Path to the root eval directory (e.g. data/eval/).

    Returns:
        Dict mapping layer int to resolved Path object.

    Raises:
        FileNotFoundError: If an expected eval file does not exist.
    """
    resolved = {}
    for layer, filename in EVAL_FILES.items():
        filepath = input_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Expected eval file for Layer {layer} not found: {filepath}"
            )
        resolved[layer] = filepath
    return resolved


# ── extraction ───────────────────────────────────────────────────────────────
def extract_questions(layer: int, filepath: Path) -> list[dict]:
    """
    Load one eval JSON file and extract question entries.

    For Layer 4 records, the `case` field is prepended to the `question`
    so the question retains its full clinical context.

    Args:
        layer:    The layer number (1–4) corresponding to this file.
        filepath: Absolute or relative path to the JSON file.

    Returns:
        List of dicts, each containing: id, layer, source, category, question.

    Raises:
        ValueError: If the file does not contain a JSON array or a record
                    is missing the required `question` field.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tolerate a nested wrapper dict (e.g. {"fullContent": [...]})
    if isinstance(data, dict) and "fullContent" in data:
        records = data["fullContent"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"Unexpected JSON structure in {filepath}: "
            "expected a top-level array or {{\"fullContent\": [...]}}."
        )

    extracted = []
    for idx, record in enumerate(records):
        validate_record(record, filepath, idx)

        question_text = build_question_text(record, layer)

        extracted.append(
            {
                "id":       record.get("id", f"layer{layer}_{idx:03d}"),
                "layer":    layer,
                "source":   record.get("source", ""),
                "category": record.get("category", ""),
                "question": question_text,
            }
        )

    return extracted


def validate_record(record: dict, filepath: Path, idx: int) -> None:
    """
    Validate that a single record contains the required `question` key.

    Args:
        record:   The dict representing one JSON record.
        filepath: Source file path (used for error messages only).
        idx:      Zero-based index of the record within the file.

    Raises:
        ValueError: If the `question` key is absent or blank.
    """
    if "question" not in record or not record["question"]:
        raise ValueError(
            f"Record at index {idx} in {filepath} is missing a 'question' field."
        )


def build_question_text(record: dict, layer: int) -> str:
    """
    Construct the final question string for a single record.

    For Layer 4 entries that carry a `case` field, the clinical scenario
    is prepended in the format:
        [CASE]
        <case text>

        [QUESTION]
        <question text>

    For all other layers the raw `question` value is returned unchanged.

    Args:
        record: The dict representing one JSON record.
        layer:  The layer number for this record.

    Returns:
        The formatted question string.
    """
    question = record["question"].strip()

    if layer == 4 and record.get("case"):
        case_text = record["case"].strip()
        return f"[CASE]\n{case_text}\n\n[QUESTION]\n{question}"

    return question


# ── output ───────────────────────────────────────────────────────────────────
def write_output(questions: list[dict], output_path: Path) -> None:
    """
    Write the consolidated question list to a JSON file.

    Each entry is annotated with a 1-based `index` field to make the
    output easy to navigate. The output directory is created if it does
    not already exist.

    Args:
        questions:   List of extracted question dicts.
        output_path: Destination file path for the JSON output.

    Raises:
        OSError: If the file cannot be written (permissions, disk full, etc.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Annotate with a 1-based sequential index
    annotated = [{"index": i + 1, **q} for i, q in enumerate(questions)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)


# ── summary ──────────────────────────────────────────────────────────────────
def print_summary(layer_counts: dict[int, int], total: int, output_path: Path) -> None:
    """
    Print a human-readable extraction summary to stdout.

    Args:
        layer_counts: Dict mapping layer number → number of questions extracted.
        total:        Total number of questions across all layers.
        output_path:  Path where the output file was written.
    """
    print(f"\n{'-' * 60}")
    print("  EXTRACTION SUMMARY")
    print(f"{'-' * 60}")
    for layer, count in sorted(layer_counts.items()):
        print(f"  Layer {layer}: {count:>4} questions")
    print(f"{'-' * 60}")
    print(f"  Total   : {total:>4} questions")
    print(f"  Output  : {output_path}")
    print(f"{'-' * 60}\n")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Entry point: parse arguments, run extraction, write output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract questions from all eval JSON files and save to "
            "a single all_questions.json file."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/eval",
        help="Path to the eval directory (default: data/eval)",
    )
    parser.add_argument(
        "--output",
        default="scripts/all_questions.json",
        help="Output file path (default: scripts/all_questions.json)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # ── validate input directory ─────────────────────────────────────────────
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nInput  : {input_dir.resolve()}")
    print(f"Output : {output_path.resolve()}")
    print(f"Excluded: baselines/ subfolder\n")

    # ── discover files ───────────────────────────────────────────────────────
    try:
        file_map = discover_json_files(input_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # ── extract ──────────────────────────────────────────────────────────────
    all_questions: list[dict] = []
    layer_counts: dict[int, int] = {}

    for layer, filepath in file_map.items():
        print(f"  [+] Layer {layer}: reading {filepath.name} ...", end=" ", flush=True)
        try:
            questions = extract_questions(layer, filepath)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"\n  [!] Error reading Layer {layer}: {e} — skipping", file=sys.stderr)
            continue

        layer_counts[layer] = len(questions)
        all_questions.extend(questions)
        print(f"{len(questions)} questions extracted")

    if not all_questions:
        print("[ERROR] No questions were extracted. Aborting.", file=sys.stderr)
        sys.exit(1)

    # ── write output ─────────────────────────────────────────────────────────
    try:
        write_output(all_questions, output_path)
    except OSError as e:
        print(f"[ERROR] Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

    # ── summary ──────────────────────────────────────────────────────────────
    print_summary(layer_counts, len(all_questions), output_path)


if __name__ == "__main__":
    main()
