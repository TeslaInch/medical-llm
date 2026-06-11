"""
ash_eval.py
───────────
Baseline evaluation script for Layer 5 — ASH 2019 SCD Guidelines
(Cardiopulmonary & Kidney Disease, reaffirmed 2025 CPKD update report).

Mirrors the architecture of baseline_eval.py but is dedicated to layer5_ASH.json,
which has a unique `rationale` field not present in other layers.

Scoring rubric (2-point scale, consistent with Layers 1 & 3):
    2 — Correct answer with sound reasoning
    1 — Partially correct or correct answer without adequate reasoning
    0 — Wrong or clinically dangerous

ASH layer specifics displayed during scoring:
  - The `rationale` field is shown after the gold answer so you can assess
    whether the model captured the correct clinical reasoning, not just the
    surface answer.
  - Source is shown per entry so the evaluator knows which ASH guideline
    sub-document the question is drawn from.

Checkpoints are saved every 4 questions (half the 8-entry set) to
data/eval/baselines/ash/ to keep them separate from the main baseline results.

Usage:
    python scripts/ash_eval.py
    python scripts/ash_eval.py --model phi3.5:mini --auto-score
    python scripts/ash_eval.py --resume 4
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama

# ── constants ─────────────────────────────────────────────────────────────────

LAYER5_PATH = Path("data/eval/layer5_ASH.json")
OUTPUT_DIR  = Path("data/eval/baselines/ash")
LAYER       = 5
MAX_SCORE   = 2

SYSTEM_PROMPT = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer clinical questions accurately and concisely, citing guideline "
    "recommendations where relevant. "
    "If you are uncertain, say so clearly rather than guessing."
)

SCORE_GUIDE = (
    "2 = correct answer + sound reasoning  |  "
    "1 = partial / correct answer, weak reasoning  |  "
    "0 = wrong or clinically dangerous"
)


# ── loader ────────────────────────────────────────────────────────────────────

def load_layer5(path: Path = LAYER5_PATH) -> list[dict]:
    """
    Load and validate the Layer 5 ASH JSON file.

    Each record is expected to contain at minimum:
    id, layer, question, answer, source, category.
    The `rationale` field is optional but displayed when present.

    Args:
        path: Path to layer5_ASH.json.

    Returns:
        List of validated question dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a JSON array or any record is missing
                    a required field.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Layer 5 file not found: {path}\n"
            "Ensure layer5_ASH.json is in data/eval/ before running this script."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}.")

    required_fields = {"id", "layer", "question", "answer", "source", "category"}
    for idx, record in enumerate(data):
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(
                f"Record at index {idx} (id={record.get('id', '?')}) "
                f"is missing required fields: {missing}"
            )
        if record.get("layer") != LAYER:
            raise ValueError(
                f"Record {record.get('id')} has layer={record.get('layer')}, "
                f"expected {LAYER}. This script is for Layer 5 only."
            )

    return data


# ── prompt builder ────────────────────────────────────────────────────────────

def build_prompt(question: dict) -> str:
    """
    Construct the prompt string sent to the model.

    Layer 5 entries have no `case` field, so the question text is sent
    directly. The source guideline is appended as context so the model
    knows this is an ASH guideline question.

    Args:
        question: A single Layer 5 record dict.

    Returns:
        Formatted prompt string.
    """
    source_context = (
        f"[Guideline context: {question['source']}]"
    )
    return f"{source_context}\n\n{question['question']}"


# ── model call ────────────────────────────────────────────────────────────────

def ask_model(model_name: str, prompt: str) -> str:
    """
    Send a prompt to the specified Ollama model and return the response text.

    Args:
        model_name: Ollama model identifier (e.g. "phi3.5:mini").
        prompt:     The user-turn prompt string.

    Returns:
        The model's response as a plain string.

    Raises:
        Exception: Any Ollama API error is propagated to the caller.
    """
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return response["message"]["content"]


# ── scorer ────────────────────────────────────────────────────────────────────

def score_entry(question: dict, response: str) -> dict:
    """
    Display the question, model response, gold answer, and rationale,
    then prompt the evaluator for an interactive score.

    The `rationale` field (unique to Layer 5) is shown after the gold answer
    so the evaluator can judge whether the model captured the correct
    clinical reasoning, not just the surface conclusion.

    Args:
        question: A single Layer 5 record dict.
        response: The model's raw response string.

    Returns:
        A result dict ready to be appended to the results list.
    """
    print("\n" + "=" * 70)
    print(f"ID       : {question['id']}")
    print(f"Category : {question['category']}")
    print(f"Source   : {question['source']}")
    print(f"\nQUESTION:\n{question['question']}")
    print(f"\nMODEL RESPONSE:\n{response}")
    print(f"\nGOLD ANSWER:\n{question['answer']}")

    rationale = question.get("rationale", "")
    if rationale:
        print(f"\nASH RATIONALE (for evaluator context):\n{rationale}")

    print(f"\n{SCORE_GUIDE}")
    score_raw = input("Score (0/1/2): ").strip()
    notes = input("Notes (optional, press Enter to skip): ").strip()

    return {
        "id":             question["id"],
        "layer":          LAYER,
        "category":       question["category"],
        "source":         question["source"],
        "question":       question["question"],
        "gold_answer":    question["answer"],
        "rationale":      rationale,
        "model_response": response,
        "score":          int(score_raw) if score_raw.isdigit() else None,
        "max_score":      MAX_SCORE,
        "notes":          notes,
    }


def auto_score_entry(question: dict, response: str) -> dict:
    """
    Build a result dict without prompting for a score (batch mode).

    Score is set to None and can be filled in during a later review pass.

    Args:
        question: A single Layer 5 record dict.
        response: The model's raw response string.

    Returns:
        A result dict with score=None and max_score=2.
    """
    return {
        "id":             question["id"],
        "layer":          LAYER,
        "category":       question["category"],
        "source":         question["source"],
        "question":       question["question"],
        "gold_answer":    question["answer"],
        "rationale":      question.get("rationale", ""),
        "model_response": response,
        "score":          None,
        "max_score":      MAX_SCORE,
        "notes":          "",
    }


# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[dict], model_name: str, suffix: str = "") -> str:
    """
    Save the current results list to a timestamped JSON file in OUTPUT_DIR.

    Args:
        results:    List of scored result dicts.
        model_name: Ollama model name (used in the filename).
        suffix:     Optional filename suffix (e.g. "checkpoint_4", "final").

    Returns:
        The absolute path of the saved file as a string.

    Raises:
        OSError: If the file cannot be written.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    safe_model = model_name.replace(":", "_").replace("/", "_")
    filename   = f"ash_{safe_model}_{timestamp}_{suffix}.json"
    path       = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return str(path)


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """
    Print a scored performance summary broken down by category.

    Only entries where score is not None are included in percentage
    calculations. Normalises each entry's score to [0, 1] before averaging.

    Args:
        results: Full list of result dicts from this session.
    """
    scored = [r for r in results if r["score"] is not None]
    if not scored:
        print("No scores recorded yet.")
        return

    pct_scores = [r["score"] / r["max_score"] for r in scored]
    overall    = sum(pct_scores) / len(pct_scores) * 100

    print(f"\n{'=' * 70}")
    print(f"  LAYER 5 — ASH EVAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model   : {results[0].get('model', 'unknown') if results else 'unknown'}")
    print(f"  Scored  : {len(scored)} / {len(results)}")
    print(f"  Overall : {overall:.1f}%")

    print("\n  By category:")
    cats = sorted(set(r["category"] for r in scored))
    for cat in cats:
        cat_results = [r for r in scored if r["category"] == cat]
        cat_pct     = (
            sum(r["score"] / r["max_score"] for r in cat_results)
            / len(cat_results)
            * 100
        )
        print(f"    {cat:<30} {cat_pct:.1f}%  ({len(cat_results)} questions)")

    print(f"{'=' * 70}\n")


# ── main runner ───────────────────────────────────────────────────────────────

def run_ash_eval(
    model_name: str,
    questions: list[dict],
    auto_score: bool = False,
    start_from: int  = 0,
) -> list[dict]:
    """
    Run the Layer 5 ASH baseline evaluation for one model.

    Iterates over all Layer 5 questions from `start_from`, calls the model,
    and either interactively scores each response or saves for later review.
    Checkpoints every 4 questions.

    Args:
        model_name:  Ollama model identifier.
        questions:   List of Layer 5 question dicts.
        auto_score:  If True, save responses without interactive scoring.
        start_from:  Zero-based index to resume from (skip earlier entries).

    Returns:
        List of result dicts for this session.
    """
    results: list[dict] = []

    print(f"\n{'=' * 70}")
    print(f"  Layer 5 — ASH Guidelines Evaluation")
    print(f"{'=' * 70}")
    print(f"  Model     : {model_name}")
    print(f"  Questions : {len(questions)}  (starting from index {start_from})")
    print(f"  Scoring   : {'batch (score later)' if auto_score else 'interactive'}")
    print(f"  Source    : ASH 2019 SCD Guidelines — CPKD (reaffirmed 2025)")
    print(f"{'=' * 70}\n")

    for i, question in enumerate(questions):
        if i < start_from:
            continue

        prompt = build_prompt(question)
        print(
            f"[{i + 1}/{len(questions)}] {question['id']} — asking model...",
            end=" ",
            flush=True,
        )

        try:
            response = ask_model(model_name, prompt)
            print("done")
        except Exception as exc:
            print(f"\n  [!] Model error: {exc} — skipping this entry")
            continue

        if auto_score:
            result = auto_score_entry(question, response)
        else:
            result = score_entry(question, response)

        result["model"] = model_name
        results.append(result)

        # Checkpoint every 4 questions
        if len(results) % 4 == 0:
            path = save_results(results, model_name, suffix=f"checkpoint_{i + 1}")
            print(f"\n  [checkpoint saved — {len(results)} questions completed]")
            print(f"  -> {path}\n")

    # Final save
    saved_path = save_results(results, model_name, suffix="final")
    print(f"\n  [final results saved -> {saved_path}]")

    print_summary(results)
    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse command-line arguments and launch the ASH evaluation run.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation on Layer 5 — ASH SCD Guidelines."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name to evaluate (e.g. phi3.5:mini). "
             "If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--auto-score",
        action="store_true",
        help="Save all responses without interactive scoring (score later).",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="INDEX",
        help="Zero-based question index to resume from (default: 0).",
    )
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading Layer 5 — ASH eval set...")
    try:
        questions = load_layer5()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(questions)} questions loaded from {LAYER5_PATH}\n")

    # ── model selection ───────────────────────────────────────────────────────
    model_name = args.model
    if not model_name:
        print("Which model do you want to evaluate?")
        print("  1. phi3.5:mini  (fast on CPU, ~3B)")
        print("  2. mistral    (slower, ~7B)")
        print("  3. custom     (type your own Ollama model name)")
        choice = input("\nChoice (1/2/3): ").strip()
        model_map = {"1": "phi3.5:mini", "2": "mistral"}
        model_name = (
            model_map[choice]
            if choice in model_map
            else input("Enter model name: ").strip()
        )

    # ── scoring mode (only prompt if not set via flag) ────────────────────────
    auto_score = args.auto_score
    if not auto_score and args.model is None:
        print("\nScoring mode:")
        print("  1. Score interactively as you go  (recommended)")
        print("  2. Save all responses first, score later")
        mode       = input("Choice (1/2): ").strip()
        auto_score = mode == "2"

    run_ash_eval(
        model_name  = model_name,
        questions   = questions,
        auto_score  = auto_score,
        start_from  = args.resume,
    )


if __name__ == "__main__":
    main()
