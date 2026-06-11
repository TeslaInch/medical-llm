"""
complete_phi3.5_mini.py
─────────────────────
Targeted re-run for phi3.5:mini on the 138 Layer 2 (MedMCQA / MedQA)
questions that were never answered due to an interrupted eval session.

Steps performed automatically:
  1. Load the master question set  (data/eval/all_questions.json)
  2. Load already-extracted answers (data/eval/phi3.5_mini_answers.json)
  3. Identify every question id that has NO answer yet.
  4. Query phi3.5:mini via Ollama for each missing question.
  5. Save a checkpoint every CHECKPOINT_EVERY questions so progress is
     never lost if the run is interrupted again.
  6. After all queries complete, merge the new answers into
     phi3.5_mini_answers.json  and write a final merged file.

Usage:
    python scripts/complete_phi3.5_mini.py

Optional flags:
    --answers-file   Path to the existing answers JSON
                     (default: data/eval/phi3.5_mini_answers.json)
    --questions-file Path to all_questions.json
                     (default: data/eval/all_questions.json)
    --output         Where to write the merged output
                     (default: data/eval/phi3.5_mini_answers.json  — overwrites in place)
    --checkpoint-dir Directory for mid-run checkpoint files
                     (default: data/eval/baselines/completion_checkpoints)
    --model          Ollama model name  (default: phi3.5:mini)
    --dry-run        Print missing question ids and exit without querying
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_NAME: str = "phi3.5:mini"
CHECKPOINT_EVERY: int = 10  # save a checkpoint file every N completions

SYSTEM_PROMPT: str = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer clinical questions accurately and concisely. "
    "If you are uncertain, say so clearly rather than guessing."
)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path, label: str) -> list | dict:
    """
    Load a JSON file and return its parsed contents.

    Args:
        path:  Absolute or relative path to the JSON file.
        label: Human-readable name used in error messages.

    Returns:
        Parsed JSON object (list or dict).

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    if not path.exists():
        print(f"[ERROR] {label} not found: {path.resolve()}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] {label} is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)


def save_json(data: list, path: Path) -> None:
    """
    Write *data* to *path* as pretty-printed JSON (UTF-8, no ASCII escaping).

    Args:
        data: List of dicts to serialise.
        path: Destination file path.  Parent directory is created if needed.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def build_prompt(question: dict) -> str:
    """
    Construct the user-facing prompt for a single question record.

    For Layer 4 records with a populated ``case`` field the clinical scenario
    is prepended to the question, matching the original baseline_eval.py
    behaviour.

    Args:
        question: A single question dict from all_questions.json.

    Returns:
        The prompt string ready to pass to the model.
    """
    case = question.get("case", "").strip()
    if case:
        return f"Clinical case:\n{case}\n\nQuestion: {question['question']}"
    return question["question"]


# ── model call ────────────────────────────────────────────────────────────────
def ask_model(model: str, prompt: str) -> str:
    """
    Send a prompt to a locally-running Ollama model and return its response.

    Args:
        model:  Ollama model identifier (e.g. ``"phi3.5:mini"``).
        prompt: The user message to send.

    Returns:
        The model's text response.

    Raises:
        Exception: Any Ollama / networking error is propagated to the caller.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return response["message"]["content"]


def build_result(question: dict, response: str, model: str) -> dict:
    """
    Build a normalised result dict from a question record and model response.

    The schema matches the existing phi3.5_mini_answers.json entries so the
    new records can be merged in cleanly.

    Args:
        question: Source question dict from all_questions.json.
        response: Raw model response text.
        model:    Model identifier string.

    Returns:
        A result dict ready for inclusion in the output file.
    """
    layer = question.get("layer", 2)
    max_score = 1 if layer == 2 else (3 if layer == 4 else 2)

    return {
        "id":             question["id"],
        "layer":          layer,
        "category":       question.get("category", ""),
        "source":         question.get("source", ""),
        "question":       question["question"],
        "model_response": response.strip(),
        "gold_answer":    question.get("answer", ""),
        "score":          None,
        "max_score":      max_score,
        "model":          model,
    }


# ── checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(results: list[dict], checkpoint_dir: Path, model: str, count: int) -> None:
    """
    Save an intermediate checkpoint file during a long run.

    Checkpoint filenames include a timestamp and the number of records
    completed so they sort naturally and are easy to identify.

    Args:
        results:        All new results collected so far this session.
        checkpoint_dir: Directory to write the checkpoint file into.
        model:          Model name (used in the filename).
        count:          Number of questions answered so far this session.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_model = model.replace(":", "_").replace("/", "_")
    filename = f"{safe_model}_completion_{timestamp}_checkpoint_{count}.json"
    path = checkpoint_dir / filename
    save_json(results, path)
    print(f"\n  [checkpoint] {count} new answers saved → {path.name}\n")


# ── merge ─────────────────────────────────────────────────────────────────────
def merge_and_reindex(existing: list[dict], new_results: list[dict]) -> list[dict]:
    """
    Merge new result records into the existing answer list, deduplicate by id,
    and re-assign 1-based sequential ``index`` values.

    If an id already exists in *existing*, the new result takes precedence
    (allows re-runs to update stale/empty responses).

    Args:
        existing:    Currently saved answer records (may already have ``index``).
        new_results: Freshly queried answer records (no ``index`` yet).

    Returns:
        The merged, sorted, re-indexed list ready to write to disk.
    """
    # Strip existing index fields so we can re-assign cleanly
    combined: dict[str, dict] = {
        r["id"]: {k: v for k, v in r.items() if k != "index"}
        for r in existing
    }
    # New results overwrite existing entries with the same id
    for r in new_results:
        clean = {k: v for k, v in r.items() if k != "index"}
        combined[r["id"]] = clean

    merged = list(combined.values())
    # Re-index with 1-based sequential numbers
    return [{"index": i + 1, **entry} for i, entry in enumerate(merged)]


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Entry point: parse arguments, find missing questions, query model, merge output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Query phi3.5:mini for the subset of questions not yet answered, "
            "then merge the new responses into phi3.5_mini_answers.json."
        )
    )
    parser.add_argument(
        "--answers-file",
        default="data/eval/phi3.5_mini_answers.json",
        help="Existing phi3.5:mini answers file (default: data/eval/phi3.5_mini_answers.json)",
    )
    parser.add_argument(
        "--questions-file",
        default="data/eval/all_questions.json",
        help="Master question list (default: data/eval/all_questions.json)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/phi3.5_mini_answers.json",
        help="Merged output path — defaults to overwriting the answers file in place",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/eval/baselines/completion_checkpoints",
        help="Directory for checkpoint files (default: data/eval/baselines/completion_checkpoints)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Ollama model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the missing question ids and exit without querying the model",
    )
    args = parser.parse_args()

    answers_path     = Path(args.answers_file)
    questions_path   = Path(args.questions_file)
    output_path      = Path(args.output)
    checkpoint_dir   = Path(args.checkpoint_dir)
    model            = args.model

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\n  Model          : {model}")
    print(f"  Answers file   : {answers_path.resolve()}")
    print(f"  Questions file : {questions_path.resolve()}")
    print(f"  Output         : {output_path.resolve()}\n")

    existing_answers: list[dict] = load_json(answers_path, "Answers file")
    all_questions: list[dict]    = load_json(questions_path, "Questions file")

    answered_ids: set[str] = {a["id"] for a in existing_answers}
    missing_questions: list[dict] = [
        q for q in all_questions if q["id"] not in answered_ids
    ]

    print(f"  Total questions      : {len(all_questions)}")
    print(f"  Already answered     : {len(answered_ids)}")
    print(f"  Missing (to query)   : {len(missing_questions)}")

    if not missing_questions:
        print("\n  ✓ All questions are already answered. Nothing to do.")
        sys.exit(0)

    # ── dry-run: just list them and exit ──────────────────────────────────────
    if args.dry_run:
        print(f"\n  Missing question ids ({len(missing_questions)}):")
        for q in missing_questions:
            print(f"    {q['id']}  (layer {q.get('layer', '?')})")
        sys.exit(0)

    # ── validate Ollama is reachable before starting ──────────────────────────
    print(f"\n  Checking Ollama connection for model '{model}'...", end=" ", flush=True)
    try:
        # Lightweight ping — ask a trivial question
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
        )
        print("OK")
    except Exception as exc:
        print(f"\n[ERROR] Cannot reach Ollama / model '{model}': {exc}", file=sys.stderr)
        print(
            "  Make sure Ollama is running  (run: ollama serve)\n"
            f"  and that the model is pulled (run: ollama pull {model})",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── query loop ────────────────────────────────────────────────────────────
    print(f"\n  Starting completion run — {len(missing_questions)} questions to go.\n")
    new_results: list[dict] = []

    for i, question in enumerate(missing_questions, start=1):
        qid    = question["id"]
        prompt = build_prompt(question)

        print(f"  [{i}/{len(missing_questions)}] {qid} — querying...", end=" ", flush=True)

        try:
            response = ask_model(model, prompt)
            print("done")
        except Exception as exc:
            print(f"\n  [!] Error on {qid}: {exc} — skipping", file=sys.stderr)
            continue

        result = build_result(question, response, model)
        new_results.append(result)

        # ── checkpoint every N completions ────────────────────────────────────
        if len(new_results) % CHECKPOINT_EVERY == 0:
            save_checkpoint(new_results, checkpoint_dir, model, len(new_results))

    # ── guard: nothing queried ────────────────────────────────────────────────
    if not new_results:
        print(
            "\n[ERROR] No new answers were collected (all queries failed).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── merge + write final output ────────────────────────────────────────────
    merged = merge_and_reindex(existing_answers, new_results)

    try:
        save_json(merged, output_path)
    except OSError as exc:
        print(f"\n[ERROR] Failed to write output: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── summary ───────────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print("  COMPLETION SUMMARY")
    print(sep)
    print(f"  Questions targeted   : {len(missing_questions)}")
    print(f"  New answers collected: {len(new_results)}")
    print(f"  Failed / skipped     : {len(missing_questions) - len(new_results)}")
    print(f"  Total in output file : {len(merged)}")
    print(sep)
    print(f"  Output               : {output_path.resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
