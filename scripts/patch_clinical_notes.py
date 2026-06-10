"""
patch_clinical_notes.py
────────────────────────
Applies targeted clinical accuracy patches to layer1_custom_notes.json
and layer3_combined.json based on verified post-2018 regulatory changes
and guideline version differences identified on 8 June 2026.

Patches applied:
  1. layer1_017 — Voxelotor withdrawal (Sept 2024) + Crizanlizumab caveat
  2. layer1_030 — Voxelotor withdrawal note in curative options answer
  3. layer1_015, layer3_012, layer3_034 — Penicillin duration guideline
                                           difference note (UK vs NIH)

Each patched entry receives:
  - clinical_note  : human-readable caveat string
  - patch_date     : ISO date of this patch
  - patch_source   : primary source URL(s) used to verify the change

Modifies files in-place. Prints a diff summary of every change made.

Usage:
    python scripts/patch_clinical_notes.py
"""

import json
import sys
from pathlib import Path

# ── file paths ────────────────────────────────────────────────────────────────
LAYER1_PATH = Path("data/eval/layer1_custom_notes.json")
LAYER3_PATH = Path("data/eval/layer3_combined.json")

PATCH_DATE = "2026-06-08"

# ── patch definitions ─────────────────────────────────────────────────────────
# Each patch is keyed by entry id.
# Fields listed here are ADDED or OVERWRITTEN on the matching entry.
# The `answer_amendment` field is NOT used to overwrite the original answer —
# it sits alongside it so the original gold answer is preserved for audit.

LAYER1_PATCHES: dict[str, dict] = {

    "layer1_017": {
        "clinical_note": (
            "POST-2018 REGULATORY UPDATE (verified 2026-06-08):\n"
            "1. VOXELOTOR (Oxbryta) — WITHDRAWN GLOBALLY. Pfizer voluntarily "
            "withdrew voxelotor from all markets on 25 September 2024 following "
            "a clinical imbalance in vaso-occlusive crises and deaths in the "
            "HOPE-KIDS 2 trial. It is no longer an available treatment option. "
            "Any model output listing voxelotor as a current SCD treatment is "
            "INCORRECT as of late 2024.\n"
            "2. CRIZANLIZUMAB (Adakveo) — EVOLVING STATUS. The confirmatory "
            "Phase III STAND trial was negative. The EMA withdrew conditional "
            "approval in 2023. As of June 2026, it remained FDA-approved in the "
            "US but its clinical role is increasingly questioned. Re-verify "
            "against the current FDA label before relying on this.\n"
            "3. GENE THERAPIES (Casgevy, Lyfgenia) — Both FDA-approved "
            "8 December 2023 for patients ≥12 years with recurrent VOCs. "
            "This predates the 2014/2018 source guidelines entirely. The "
            "mention of these in the gold answer is CORRECT and current.\n"
            "4. L-GLUTAMINE (Endari) — Approved 2017, still available. "
            "No change."
        ),
        "answer_amendment": (
            "AMENDED: Remove voxelotor from the list of currently available "
            "treatments — it was withdrawn globally in September 2024. "
            "Crizanlizumab's status is contested (STAND trial negative, "
            "EMA withdrawn); flag as uncertain for clinical use. "
            "Casgevy and Lyfgenia (approved Dec 2023) remain correct."
        ),
        "patch_date": PATCH_DATE,
        "patch_source": (
            "FDA Drug Safety Communications; Pfizer press release "
            "25 Sep 2024 (voxelotor withdrawal); EMA product page "
            "(crizanlizumab conditional MA withdrawal 2023); "
            "FDA approval letters for Casgevy and Lyfgenia, 8 Dec 2023."
        ),
    },

    "layer1_030": {
        "clinical_note": (
            "POST-2018 UPDATE (verified 2026-06-08):\n"
            "VOXELOTOR — WITHDRAWN. Pfizer withdrew voxelotor globally on "
            "25 September 2024. It should not be listed as a curative or "
            "disease-modifying option.\n"
            "GENE THERAPIES — Casgevy (CRISPR/Cas9, editing BCL11A enhancer) "
            "and Lyfgenia (lentiviral; carries a boxed warning for haematologic "
            "malignancy) were both FDA-approved 8 December 2023 for patients "
            "≥12 years with recurrent VOCs. The gold answer correctly names both. "
            "Cost remains a major access barrier in Nigeria (~$2–3M per treatment "
            "as of 2025)."
        ),
        "answer_amendment": (
            "AMENDED: Remove voxelotor from curative/newer options — withdrawn "
            "Sept 2024. Casgevy and Lyfgenia remain correct."
        ),
        "patch_date": PATCH_DATE,
        "patch_source": (
            "Pfizer press release 25 Sep 2024; "
            "FDA approval letters Casgevy/Lyfgenia 8 Dec 2023."
        ),
    },

    "layer1_015": {
        "clinical_note": (
            "GUIDELINE VERSION DIFFERENCE (not an error — both are correct "
            "for their respective guideline):\n"
            "NHLBI 2014 → Penicillin prophylaxis can be DISCONTINUED at age 5 "
            "in children with no history of splenectomy or invasive pneumococcal "
            "infection.\n"
            "UK Standards 2018 (adult-focused) → Lifelong penicillin prophylaxis "
            "is recommended for all adults with HbSS and HbSβ⁰ due to permanent "
            "functional asplenia.\n"
            "This is a genuine policy difference between the two guidelines, not "
            "a contradiction. The gold answer reflects the NHLBI 2014 position. "
            "If deploying for a UK clinical audience, the UK 2018 position applies."
        ),
        "patch_date": PATCH_DATE,
        "patch_source": (
            "NHLBI 2014 Expert Panel Report, p.15; "
            "UK Standards for Clinical Care of Adults with SCD (2018), p.31."
        ),
    },

}

LAYER3_PATCHES: dict[str, dict] = {

    "layer3_012": {
        "clinical_note": (
            "GUIDELINE VERSION DIFFERENCE (not an error — both are correct "
            "for their respective guideline):\n"
            "UK Standards 2018 (this source) → Lifelong penicillin for all "
            "HbSS/HbSβ⁰ adults.\n"
            "NHLBI 2014 → Can stop at age 5 (paediatric guidance). "
            "These guidelines cover different populations (adult vs general). "
            "A model trained on both may appear self-contradictory but "
            "both answers are clinically correct in context."
        ),
        "patch_date": PATCH_DATE,
        "patch_source": (
            "UK Standards for Clinical Care of Adults with SCD (2018), p.31; "
            "NHLBI 2014 Expert Panel Report, p.15."
        ),
    },

    "layer3_034": {
        "clinical_note": (
            "GUIDELINE VERSION DIFFERENCE (not an error):\n"
            "NHLBI 2014 (this source) → Prophylactic penicillin can be "
            "discontinued at age 5 in the absence of splenectomy or invasive "
            "pneumococcal disease.\n"
            "UK Standards 2018 → Lifelong prophylaxis for HbSS/HbSβ⁰ adults.\n"
            "Both are correct within their guideline context."
        ),
        "patch_date": PATCH_DATE,
        "patch_source": (
            "NHLBI 2014 Expert Panel Report, p.15; "
            "UK Standards for Clinical Care of Adults with SCD (2018), p.31."
        ),
    },

}


# ── helpers ───────────────────────────────────────────────────────────────────

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
        raise ValueError(f"Expected a JSON array in {path}")
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


def apply_patches(
    records: list[dict],
    patches: dict[str, dict],
    label: str,
) -> int:
    """
    Apply a set of patches to matching records by id.

    Each patch dict is merged into the matching record. Existing fields
    are preserved; only fields listed in the patch are added or updated.

    Args:
        records: List of record dicts (mutated in-place).
        patches: Dict mapping record id → patch fields to apply.
        label:   Human-readable layer name for console output.

    Returns:
        Number of records successfully patched.
    """
    patched = 0
    patch_ids = set(patches.keys())

    for record in records:
        rid = record.get("id")
        if rid in patch_ids:
            patch = patches[rid]
            for key, value in patch.items():
                record[key] = value
            print(f"  [PATCHED] {label} / {rid}")
            patched += 1

    unmatched = patch_ids - {r.get("id") for r in records}
    for uid in sorted(unmatched):
        print(f"  [WARNING] Patch target not found: {label} / {uid}", file=sys.stderr)

    return patched


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: load layer files, apply patches, save in-place.
    """
    print("\nLoading files...")

    try:
        layer1 = load_json(LAYER1_PATH)
        layer3 = load_json(LAYER3_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Layer 1: {len(layer1)} entries")
    print(f"  Layer 3: {len(layer3)} entries")

    print("\nApplying patches...")
    l1_patched = apply_patches(layer1, LAYER1_PATCHES, "Layer 1")
    l3_patched = apply_patches(layer3, LAYER3_PATCHES, "Layer 3")

    print("\nSaving...")
    try:
        save_json(layer1, LAYER1_PATH)
        print(f"  Saved -> {LAYER1_PATH}")
        save_json(layer3, LAYER3_PATH)
        print(f"  Saved -> {LAYER3_PATH}")
    except OSError as e:
        print(f"[ERROR] Could not save: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'-' * 60}")
    print("  PATCH SUMMARY")
    print(f"{'-' * 60}")
    print(f"  Layer 1 entries patched : {l1_patched}")
    print(f"  Layer 3 entries patched : {l3_patched}")
    print(f"  Total                   : {l1_patched + l3_patched}")
    print(f"{'-' * 60}\n")


if __name__ == "__main__":
    main()
