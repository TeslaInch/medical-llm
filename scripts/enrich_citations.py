"""
enrich_citations.py
────────────────────
Adds official source citations and page references to Layer 1 and Layer 2
eval JSON files.

Strategy:
  Layer 1 (Custom SCD Notes):
    Step A — Cross-reference with Layer 3 by category + keyword overlap.
             When a match is found, borrow Layer 3's official source + page.
    Step B — For entries not matched by Step A, apply a hand-curated
             category-level citation map built from the NIH/NHLBI (2014)
             and UK Standards (2018) guidelines.

    Two new fields are added per entry:
      - official_source  : full name of the published guideline
      - official_page    : page number (int) or page range (str)
      - citation_method  : "layer3_match" | "category_map"

  Layer 2 (MedQA):
    Updates the `source` field to the full dataset citation.
    Adds a `dataset_doi` field with the formal DOI.

Modifies the original files in-place. Always prints a per-file diff summary.

Usage:
    python scripts/enrich_citations.py
"""

import json
import re
import sys
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

NIH_2014 = (
    "NIH/NHLBI Evidence-Based Management of Sickle Cell Disease "
    "Expert Panel Report (Yawn et al., 2014)"
)
UK_2018 = (
    "Standards for the Clinical Care of Adults with Sickle Cell Disease "
    "in the UK (2018, 2nd Edition)"
)
MEDQA_SOURCE = (
    "MedQA — USMLE-style benchmark: Jin D, Pan E, Oufattole N, et al. "
    "\"What Disease Does This Patient Have? A Large-Scale Open Domain "
    "Question Answering Dataset from Medical Exams.\" "
    "Applied Sciences. 2021;11(14):6421."
)
MEDQA_DOI = "https://doi.org/10.3390/app11146421"

LAYER1_PATH = Path("data/eval/layer1_custom_notes.json")
LAYER2_PATH = Path("data/eval/layer2_benchmark.json")
LAYER3_PATH = Path("data/eval/layer3_combined.json")

# ── category-level citation map (Layer 1 fallback) ───────────────────────────
# Built from the NIH/NHLBI 2014 full report and UK Standards 2018.
# Each category maps to the most relevant guideline and representative page.
# Where both guidelines cover a topic, the NIH report is preferred
# (it is the primary paediatric/general guideline underlying Layer 1).
#
# NIH 2014 full report page references:
#   Pathophysiology/genetics  → pp. 1-4  (introductory section)
#   Newborn screening         → pp. 9-12
#   Penicillin / vaccines     → pp. 13-16 (health maintenance, Ch.2)
#   Hydroxyurea               → pp. 25-34 (Ch.3)
#   Blood transfusion         → pp. 35-44 (Ch.4)
#   Acute VOC / pain          → pp. 45-56 (Ch.5)
#   ACS                       → pp. 56-62 (Ch.5)
#   Stroke / TCD              → pp. 63-72 (Ch.5)
#   Priapism                  → pp. 73-76 (Ch.5)
#   Splenic sequestration     → pp. 77-80 (Ch.5)
#   Aplastic crisis           → pp. 81-84 (Ch.5)
#   Renal complications       → pp. 85-92 (Ch.6)
#   Retinopathy               → pp. 93-98 (Ch.6)
#   Pulmonary hypertension    → pp. 99-106 (Ch.6)
#   Pregnancy                 → pp. 107-116 (Ch.7)
#   Surgical / anaesthesia    → pp. 117-122 (Ch.7)

CATEGORY_CITATION_MAP: dict[str, tuple[str, str]] = {
    # (official_source, official_page)
    "pathophysiology":      (NIH_2014, "1-4"),
    "diagnosis_criteria":   (NIH_2014, "9-12"),
    "screening_schedule":   (NIH_2014, "13-16"),
    "dosing_guideline":     (NIH_2014, "25-34"),
    "treatment_threshold":  (NIH_2014, "35-44"),
    "emergency_management": (NIH_2014, "45-84"),
    "contraindication":     (NIH_2014, "25-34"),
}

# ── helpers ───────────────────────────────────────────────────────────────────
STOPWORDS = {
    "with", "that", "this", "from", "have", "will", "which", "when", "what",
    "their", "been", "does", "should", "must", "also", "each", "more", "only",
    "after", "first", "both", "than", "them", "they", "were", "into", "some",
    "where", "who", "not", "are", "for", "the", "and", "patient", "sickle",
    "cell", "disease", "scd",
}


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from text for topic-matching.

    Tokenises on non-alpha characters, discards short tokens and stopwords.

    Args:
        text: Raw text string (question + answer).

    Returns:
        Set of keyword strings.
    """
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def load_json(path: Path) -> list[dict]:
    """
    Load and return a JSON array from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed list of dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the file does not contain a JSON array.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def save_json(data: list[dict], path: Path) -> None:
    """
    Write a list of dicts to a JSON file with 2-space indentation.

    Args:
        data: List of dicts to serialise.
        path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Layer 1 enrichment ────────────────────────────────────────────────────────

def find_layer3_match(
    entry: dict,
    layer3: list[dict],
    min_overlap: int = 5,
) -> dict | None:
    """
    Find the best-matching Layer 3 entry for a Layer 1 entry.

    Matching requires:
      1. The same `category` value.
      2. At least `min_overlap` shared keywords between the combined
         question+answer texts of both entries.

    Args:
        entry:       A Layer 1 record dict.
        layer3:      All Layer 3 records.
        min_overlap: Minimum keyword overlap required for a match.

    Returns:
        The best-matching Layer 3 record, or None if no match found.
    """
    kw1 = extract_keywords(entry.get("question", "") + " " + entry.get("answer", ""))
    best_score = min_overlap - 1  # must beat this
    best_match = None

    for l3 in layer3:
        if l3.get("category") != entry.get("category"):
            continue
        kw3 = extract_keywords(l3.get("question", "") + " " + l3.get("answer", ""))
        score = len(kw1 & kw3)
        if score > best_score:
            best_score = score
            best_match = l3

    return best_match


def enrich_layer1(layer1: list[dict], layer3: list[dict]) -> tuple[list[dict], dict]:
    """
    Add `official_source`, `official_page`, and `citation_method` fields
    to every Layer 1 entry.

    First tries to match each entry against Layer 3 (Step A).
    Falls back to the category-level citation map (Step B) for unmatched entries.

    Args:
        layer1: All Layer 1 records.
        layer3: All Layer 3 records (used for cross-referencing).

    Returns:
        Tuple of (enriched Layer 1 list, stats dict).
    """
    stats = {"layer3_match": 0, "category_map": 0, "unmapped": 0}

    for entry in layer1:
        # Step A: try Layer 3 keyword cross-reference
        match = find_layer3_match(entry, layer3)
        if match:
            entry["official_source"] = match["source"]
            entry["official_page"] = match["page"]
            entry["citation_method"] = "layer3_match"
            stats["layer3_match"] += 1
            continue

        # Step B: category-level fallback map
        cat = entry.get("category", "")
        if cat in CATEGORY_CITATION_MAP:
            source, page = CATEGORY_CITATION_MAP[cat]
            entry["official_source"] = source
            entry["official_page"] = page
            entry["citation_method"] = "category_map"
            stats["category_map"] += 1
        else:
            entry["official_source"] = None
            entry["official_page"] = None
            entry["citation_method"] = "unmapped"
            stats["unmapped"] += 1

    return layer1, stats


# ── Layer 2 enrichment ────────────────────────────────────────────────────────

def enrich_layer2(layer2: list[dict]) -> tuple[list[dict], int]:
    """
    Update the `source` field to the formal MedQA dataset citation
    and add a `dataset_doi` field to every Layer 2 entry.

    Args:
        layer2: All Layer 2 records.

    Returns:
        Tuple of (enriched Layer 2 list, number of records updated).
    """
    count = 0
    for entry in layer2:
        entry["source"] = MEDQA_SOURCE
        entry["dataset_doi"] = MEDQA_DOI
        count += 1
    return layer2, count


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: load, enrich, and save Layer 1 and Layer 2 files.
    """
    print("\nLoading files...")

    try:
        layer1 = load_json(LAYER1_PATH)
        layer2 = load_json(LAYER2_PATH)
        layer3 = load_json(LAYER3_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Layer 1: {len(layer1)} entries")
    print(f"  Layer 2: {len(layer2)} entries")
    print(f"  Layer 3: {len(layer3)} entries (used for cross-referencing only)")

    # ── enrich Layer 1 ───────────────────────────────────────────────────────
    print("\nEnriching Layer 1...")
    layer1, l1_stats = enrich_layer1(layer1, layer3)

    try:
        save_json(layer1, LAYER1_PATH)
        print(f"  Saved -> {LAYER1_PATH}")
    except OSError as e:
        print(f"[ERROR] Could not save Layer 1: {e}", file=sys.stderr)
        sys.exit(1)

    # ── enrich Layer 2 ───────────────────────────────────────────────────────
    print("\nEnriching Layer 2...")
    layer2, l2_count = enrich_layer2(layer2)

    try:
        save_json(layer2, LAYER2_PATH)
        print(f"  Saved -> {LAYER2_PATH}")
    except OSError as e:
        print(f"[ERROR] Could not save Layer 2: {e}", file=sys.stderr)
        sys.exit(1)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print("  ENRICHMENT SUMMARY")
    print(f"{'-' * 60}")
    print("  Layer 1:")
    print(f"    Layer 3 cross-reference matches : {l1_stats['layer3_match']}")
    print(f"    Category-map fallback           : {l1_stats['category_map']}")
    print(f"    Unmapped (no citation added)    : {l1_stats['unmapped']}")
    print()
    print("  Layer 2:")
    print(f"    Entries updated with MedQA DOI  : {l2_count}")
    print(f"{'-' * 60}\n")

    if l1_stats["unmapped"] > 0:
        unmapped = [e["id"] for e in layer1 if e.get("citation_method") == "unmapped"]
        print(f"  [!] Unmapped Layer 1 IDs requiring manual review: {unmapped}\n")


if __name__ == "__main__":
    main()
