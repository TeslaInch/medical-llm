"""
score_comparison.py
────────────────────
Side-by-side evaluation of phi3:mini vs Claude Opus 4.8 on the SCD benchmark.

Both models are scored against the same gold standard answers using identical
layer-specific rubrics. This script is the single place where all scoring
happens — do not pre-score either model separately, as that introduces
evaluator drift.

Coverage summary (auto-detected at runtime):
  phi3:mini   — Layers 1–4 (merged from all checkpoint files) + Layer 5 (baselines/ash/)
  Claude Opus — Layers 1–5 (SCD_Answer_Key_ALL.json)

Scoring methods:
  Layer 2 (MCQ)  — Automatic keyword match: gold answer text found in response?
                   Identical algorithm applied to both models. No human needed.
  Layers 1,3,4,5 — Human side-by-side: both responses shown together,
                   both scored in the same sitting, same rubric, same evaluator.

Output:
  data/eval/comparison/comparison_results_<timestamp>.json  — per-question detail
  data/eval/comparison/comparison_summary_<timestamp>.json  — aggregate by layer/category

Usage:
    python scripts/score_comparison.py
    python scripts/score_comparison.py --layers 1,3,4,5    # skip auto-scored L2
    python scripts/score_comparison.py --auto-only          # only run L2 auto-scoring
    python scripts/score_comparison.py --resume data/eval/comparison/comparison_results_<ts>.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
EVAL_FILES = {
    1: Path("data/eval/layer1_custom_notes.json"),
    2: Path("data/eval/layer2_benchmark.json"),
    3: Path("data/eval/layer3_combined.json"),
    4: Path("data/eval/layer4_clinical_cases.json"),
    5: Path("data/eval/layer5_ASH.json"),
}
BASELINES_DIR   = Path("data/eval/baselines")
ASH_DIR         = BASELINES_DIR / "ash"
CLAUDE_FILE     = Path("data/eval/SCD_Answer_Key_ALL.json")
OUTPUT_DIR      = Path("data/eval/comparison")

# ── scoring rubrics ────────────────────────────────────────────────────────────
RUBRICS = {
    1: "2 = correct + sound reasoning  |  1 = partial / weak reasoning  |  0 = wrong",
    2: "1 = correct  |  0 = wrong  [AUTO-SCORED via keyword match]",
    3: "2 = correct + sound reasoning  |  1 = partial / weak reasoning  |  0 = wrong",
    4: "3 = complete (diagnosis + full management)  |  2 = mostly correct  |  1 = partial  |  0 = wrong",
    5: "2 = correct + sound reasoning  |  1 = partial / weak reasoning  |  0 = wrong",
}
MAX_SCORES = {1: 2, 2: 1, 3: 2, 4: 3, 5: 2}


# ── data loaders ───────────────────────────────────────────────────────────────

def load_json_file(path: Path) -> list[dict]:
    """
    Load and return a JSON array from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed list of dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a JSON array.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_gold_standard() -> dict[str, dict]:
    """
    Load the gold standard answers from all layer eval files.

    Returns:
        Dict mapping question id → record dict (with `answer`, `layer`,
        `category`, `question`, `source`, and optional `case`/`rationale`).
    """
    gold: dict[str, dict] = {}
    for layer, path in EVAL_FILES.items():
        if not path.exists():
            print(f"  [!] Gold standard for Layer {layer} not found: {path}")
            continue
        records = load_json_file(path)

        # Layer 1 may be wrapped in {"fullContent": [...]}
        if isinstance(records, dict) and "fullContent" in records:
            records = records["fullContent"]

        for r in records:
            rid = r.get("id")
            if rid:
                gold[rid] = {
                    "id":        rid,
                    "layer":     layer,
                    "category":  r.get("category", ""),
                    "source":    r.get("source", ""),
                    "question":  r.get("question", ""),
                    "case":      r.get("case", ""),
                    "rationale": r.get("rationale", ""),
                    "gold_answer": r.get("answer", ""),
                    "max_score": MAX_SCORES[layer],
                }
    return gold


def load_phi3_responses() -> dict[str, str]:
    """
    Merge all phi3:mini checkpoint files (Layers 1–4) and the ASH final file
    (Layer 5) into a single dict of id → model_response.

    The latest checkpoint file wins for any duplicated id — this ensures the
    most recent model state is used for each question.

    Returns:
        Dict mapping question id → phi3 response string.
    """
    merged: dict[str, dict] = {}

    # Layers 1–4: all checkpoint + final files in baselines/
    for fpath in sorted(BASELINES_DIR.glob("phi3_mini_*.json")):
        records = load_json_file(fpath)
        for r in records:
            rid = r.get("id")
            if rid:
                merged[rid] = r  # later files overwrite earlier ones

    # Layer 5: ASH final file
    ash_finals = sorted(ASH_DIR.glob("*_final.json"))
    if ash_finals:
        records = load_json_file(ash_finals[-1])
        for r in records:
            rid = r.get("id")
            if rid:
                merged[rid] = r

    return {rid: r.get("model_response", "") for rid, r in merged.items()}


def load_claude_responses() -> dict[str, str]:
    """
    Load Claude Opus 4.8 responses from SCD_Answer_Key_ALL.json.

    Returns:
        Dict mapping question id → Claude response string.

    Raises:
        FileNotFoundError: If SCD_Answer_Key_ALL.json is not found.
    """
    records = load_json_file(CLAUDE_FILE)
    return {r["id"]: r.get("answer", "") for r in records if "id" in r}


# ── alignment ──────────────────────────────────────────────────────────────────

def align_data(
    gold: dict[str, dict],
    phi3: dict[str, str],
    claude: dict[str, str],
    layers: list[int],
) -> list[dict]:
    """
    Build a per-question alignment table covering all requested layers.

    Questions where a model has no response are recorded as "[NO RESPONSE]"
    rather than being excluded or scored as 0 — this preserves honesty about
    coverage gaps.

    Args:
        gold:   Gold standard records keyed by id.
        phi3:   phi3 model responses keyed by id.
        claude: Claude responses keyed by id.
        layers: List of layer numbers to include.

    Returns:
        List of aligned question dicts, ready for scoring.
    """
    aligned = []
    for rid, g in gold.items():
        if g["layer"] not in layers:
            continue
        aligned.append({
            "id":           rid,
            "layer":        g["layer"],
            "category":     g["category"],
            "source":       g["source"],
            "question":     g["question"],
            "case":         g.get("case", ""),
            "rationale":    g.get("rationale", ""),
            "gold_answer":  g["gold_answer"],
            "max_score":    g["max_score"],
            "phi3_response":   phi3.get(rid, "[NO RESPONSE]"),
            "claude_response": claude.get(rid, "[NO RESPONSE]"),
            # Scores filled in during evaluation
            "phi3_score":      None,
            "claude_score":    None,
            "scoring_method":  None,
            "notes":           "",
        })

    # Sort by layer then id for a consistent evaluation order
    aligned.sort(key=lambda x: (x["layer"], x["id"]))
    return aligned


# ── Layer 2 auto-scoring ───────────────────────────────────────────────────────

def keyword_match(gold_answer: str, model_response: str) -> int:
    """
    Score a Layer 2 MCQ response by checking if the gold answer keyword
    appears anywhere in the model response (case-insensitive).

    The gold answer is normalised to its core term (stripped of surrounding
    punctuation) before matching, so short answers like "100%" or
    "P. falciparum" are matched correctly even if the response contains them
    embedded in a longer sentence.

    Args:
        gold_answer:    The correct answer string from the eval file.
        model_response: The model's free-text response.

    Returns:
        1 if the gold answer is found in the response, 0 otherwise.
    """
    if not gold_answer or model_response == "[NO RESPONSE]":
        return 0

    # Normalise: lowercase, strip surrounding quotes/asterisks
    gold_norm    = gold_answer.strip().lower().strip("*\"'`")
    response_norm = model_response.lower()

    # Direct substring match
    if gold_norm in response_norm:
        return 1

    # Fallback: match the first meaningful word (handles "P. falciparum" → "falciparum")
    core = re.sub(r"[^a-z0-9%\.\-]", " ", gold_norm).strip().split()
    if core and any(word in response_norm for word in core if len(word) > 3):
        return 1

    return 0


def auto_score_layer2(questions: list[dict]) -> list[dict]:
    """
    Apply keyword-match auto-scoring to all Layer 2 questions in-place.

    Sets `phi3_score`, `claude_score`, and `scoring_method` = "auto_keyword"
    for each Layer 2 entry.

    Args:
        questions: Full aligned question list (mutated in-place).

    Returns:
        The same list with Layer 2 scores filled in.
    """
    for q in questions:
        if q["layer"] != 2:
            continue
        q["phi3_score"]   = keyword_match(q["gold_answer"], q["phi3_response"])
        q["claude_score"] = keyword_match(q["gold_answer"], q["claude_response"])
        q["scoring_method"] = "auto_keyword"
    return questions


# ── human side-by-side scoring ─────────────────────────────────────────────────

def display_question(q: dict, i: int, total: int) -> None:
    """
    Print a fully formatted question card for human evaluation.

    Shows: question (with case if Layer 4), gold answer, ASH rationale
    (if Layer 5), then both model responses side by side.

    Args:
        q:     Aligned question dict.
        i:     1-based position in the scoring queue.
        total: Total questions in the queue.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  [{i}/{total}]  {q['id']}  |  Layer {q['layer']}  |  {q['category']}")
    print(sep)

    if q.get("case"):
        print(f"\nCASE:\n{q['case']}")

    print(f"\nQUESTION:\n{q['question']}")
    print(f"\nGOLD ANSWER:\n{q['gold_answer']}")

    if q.get("rationale"):
        print(f"\nASH RATIONALE (evaluator context):\n{q['rationale']}")

    print(f"\n{'-' * 36}  phi3:mini  {'-' * 36}")
    phi3_text = q["phi3_response"]
    if phi3_text == "[NO RESPONSE]":
        print("  [NO RESPONSE — not covered in baseline]")
    else:
        print(phi3_text[:1200] + ("..." if len(phi3_text) > 1200 else ""))

    print(f"\n{'-' * 33}  Claude Opus 4.8  {'-' * 33}")
    claude_text = q["claude_response"]
    if claude_text == "[NO RESPONSE]":
        print("  [NO RESPONSE]")
    else:
        print(claude_text[:1200] + ("..." if len(claude_text) > 1200 else ""))

    print(f"\nRUBRIC: {RUBRICS[q['layer']]}")


def human_score_entry(q: dict, i: int, total: int) -> dict:
    """
    Display the question card and prompt for interactive scores for both models.

    If a model has [NO RESPONSE], its score is automatically set to 0 and
    no prompt is shown for it.

    Args:
        q:     Aligned question dict (mutated in-place with scores).
        i:     1-based position in the scoring queue.
        total: Total questions in the queue.

    Returns:
        The updated question dict with scores and notes filled in.
    """
    display_question(q, i, total)
    max_s = q["max_score"]

    # Score phi3
    if q["phi3_response"] == "[NO RESPONSE]":
        q["phi3_score"] = 0
        print("\nphi3:mini score: 0 (no response)")
    else:
        raw = input(f"\nScore phi3:mini (0–{max_s}): ").strip()
        q["phi3_score"] = int(raw) if raw.isdigit() else None

    # Score Claude
    if q["claude_response"] == "[NO RESPONSE]":
        q["claude_score"] = 0
        print("Claude score: 0 (no response)")
    else:
        raw = input(f"Score Claude Opus 4.8 (0–{max_s}): ").strip()
        q["claude_score"] = int(raw) if raw.isdigit() else None

    q["notes"] = input("Notes (optional, Enter to skip): ").strip()
    q["scoring_method"] = "human"
    return q


# ── persistence ────────────────────────────────────────────────────────────────

def save_results(results: list[dict], timestamp: str, suffix: str = "results") -> str:
    """
    Write the current results list to a timestamped JSON file.

    Args:
        results:   List of scored question dicts.
        timestamp: Timestamp string used in the filename.
        suffix:    File suffix ("results" or "summary").

    Returns:
        Absolute path of the saved file as a string.

    Raises:
        OSError: If the directory cannot be created or file cannot be written.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"comparison_{suffix}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return str(path)


def build_summary(results: list[dict]) -> dict:
    """
    Compute aggregate scoring statistics from the results list.

    Statistics are broken down by overall, by layer, and by category.
    Only entries where both scores are non-None are included in percentages.

    Args:
        results: Full list of scored result dicts.

    Returns:
        Summary dict with "overall", "by_layer", "by_category" keys.
    """
    def pct(score, max_s):
        """Calculate percentage score normalised to max_score."""
        return (score / max_s * 100) if max_s > 0 else 0

    # Filter to fully scored entries
    scored = [r for r in results if r["phi3_score"] is not None and r["claude_score"] is not None]

    def aggregate(entries):
        """Return aggregate stats dict for a group of entries."""
        if not entries:
            return {"count": 0, "phi3_pct": None, "claude_pct": None, "winner": None}
        phi3_pcts   = [pct(r["phi3_score"],   r["max_score"]) for r in entries]
        claude_pcts = [pct(r["claude_score"], r["max_score"]) for r in entries]
        phi3_avg    = sum(phi3_pcts)   / len(phi3_pcts)
        claude_avg  = sum(claude_pcts) / len(claude_pcts)
        winner = (
            "phi3:mini"      if phi3_avg > claude_avg else
            "claude_opus_4.8" if claude_avg > phi3_avg else
            "tie"
        )
        return {
            "count":      len(entries),
            "phi3_pct":   round(phi3_avg,   1),
            "claude_pct": round(claude_avg, 1),
            "winner":     winner,
        }

    # By layer
    by_layer = {}
    for layer in sorted(set(r["layer"] for r in results)):
        by_layer[f"layer{layer}"] = aggregate(
            [r for r in scored if r["layer"] == layer]
        )

    # By category
    by_cat = {}
    for cat in sorted(set(r["category"] for r in results)):
        by_cat[cat] = aggregate(
            [r for r in scored if r["category"] == cat]
        )

    return {
        "generated_at":   datetime.now().isoformat(),
        "total_questions": len(results),
        "scored":         len(scored),
        "unscored":       len(results) - len(scored),
        "overall":        aggregate(scored),
        "by_layer":       by_layer,
        "by_category":    by_cat,
    }


def print_summary(summary: dict) -> None:
    """
    Print a formatted comparison summary to stdout.

    Args:
        summary: Summary dict produced by build_summary().
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  COMPARISON SUMMARY — phi3:mini  vs  Claude Opus 4.8")
    print(sep)
    o = summary["overall"]
    print(f"  Scored   : {summary['scored']} / {summary['total_questions']}")
    if o["count"]:
        print(f"  OVERALL  :  phi3 {o['phi3_pct']:.1f}%  |  Claude {o['claude_pct']:.1f}%  |  Winner: {o['winner']}")

    print("\n  By layer:")
    for key, stats in summary["by_layer"].items():
        if stats["count"]:
            print(f"    {key:<10}  phi3 {stats['phi3_pct']:>5.1f}%  |  Claude {stats['claude_pct']:>5.1f}%  |  {stats['winner']}")

    print("\n  By category:")
    for cat, stats in summary["by_category"].items():
        if stats["count"]:
            print(f"    {cat:<30}  phi3 {stats['phi3_pct']:>5.1f}%  |  Claude {stats['claude_pct']:>5.1f}%")

    print(sep + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: parse arguments, load data, run scoring, save results.
    """
    parser = argparse.ArgumentParser(
        description="Side-by-side scoring of phi3:mini vs Claude Opus 4.8 on the SCD benchmark."
    )
    parser.add_argument(
        "--layers",
        default="1,2,3,4,5",
        help="Comma-separated layer numbers to score (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Only run Layer 2 auto-scoring; skip human scoring for other layers.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a previous comparison_results_*.json file to resume from.",
    )
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip().isdigit()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ── load ──────────────────────────────────────────────────────────────────
    print("\nLoading gold standard...")
    try:
        gold = load_gold_standard()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(gold)} gold standard answers loaded")

    print("Loading phi3:mini responses...")
    phi3 = load_phi3_responses()
    print(f"  {len(phi3)} phi3 responses loaded")

    print("Loading Claude Opus 4.8 responses...")
    try:
        claude = load_claude_responses()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(claude)} Claude responses loaded")

    # ── align ─────────────────────────────────────────────────────────────────
    questions = align_data(gold, phi3, claude, layers)
    print(f"\nAligned {len(questions)} questions across layers {layers}")

    # ── resume ────────────────────────────────────────────────────────────────
    scored_ids: set[str] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            previous = load_json_file(resume_path)
            prev_map = {r["id"]: r for r in previous}
            for q in questions:
                if q["id"] in prev_map and prev_map[q["id"]].get("scoring_method"):
                    q.update(prev_map[q["id"]])
                    scored_ids.add(q["id"])
            print(f"  Resumed: {len(scored_ids)} questions already scored, {len(questions) - len(scored_ids)} remaining")
        else:
            print(f"  [!] Resume file not found: {resume_path}")

    # ── Layer 2 auto-scoring ──────────────────────────────────────────────────
    if 2 in layers:
        l2 = [q for q in questions if q["layer"] == 2 and q["id"] not in scored_ids]
        if l2:
            print(f"\nAuto-scoring {len(l2)} Layer 2 MCQ questions...")
            auto_score_layer2(questions)
            for q in questions:
                if q["layer"] == 2:
                    scored_ids.add(q["id"])
            print(f"  Done.")

    # ── save after auto-scoring ───────────────────────────────────────────────
    save_results(questions, timestamp)

    if args.auto_only:
        summary = build_summary(questions)
        print_summary(summary)
        save_results(summary, timestamp, suffix="summary")
        return

    # ── human scoring for remaining layers ───────────────────────────────────
    to_score = [q for q in questions if q["layer"] != 2 and q["id"] not in scored_ids]
    print(f"\n{len(to_score)} questions require human scoring (Layers 1, 3, 4, 5)")
    if not to_score:
        print("  Nothing left to score — generating summary.")
    else:
        print("  Press Ctrl+C at any time to stop; progress is checkpointed every 10 questions.\n")

    for i, q in enumerate(to_score, start=1):
        try:
            human_score_entry(q, i, len(to_score))
            scored_ids.add(q["id"])
        except KeyboardInterrupt:
            print("\n\n  [Interrupted] Saving progress...")
            break
        except Exception as exc:
            print(f"\n  [!] Error scoring {q['id']}: {exc} — skipping")
            continue

        # Checkpoint every 10 questions
        if i % 10 == 0:
            save_results(questions, timestamp)
            print(f"\n  [checkpoint — {i} human-scored this session]\n")

    # ── final save + summary ──────────────────────────────────────────────────
    results_path = save_results(questions, timestamp)
    summary = build_summary(questions)
    summary_path = save_results(summary, timestamp, suffix="summary")

    print_summary(summary)
    print(f"  Results saved  -> {results_path}")
    print(f"  Summary saved  -> {summary_path}\n")


if __name__ == "__main__":
    main()
