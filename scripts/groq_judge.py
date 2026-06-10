"""
groq_judge.py
─────────────
LLM-as-Judge scorer for the SCD benchmark.

Uses the Groq API (llama-3.3-70b-versatile) to score phi3:mini and Claude
Opus 4.8 responses against gold-standard answers for Layers 1, 3, 4, and 5.
Layer 2 (MCQ) is handled by keyword-match auto-scoring — identical to the
existing score_comparison.py logic — and is NOT sent to the judge.

Design choices:
  - Separate API calls per model: phi3 and Claude are judged independently
    so the judge cannot anchor on whichever response it sees first.
  - Temperature = 0: fully deterministic scoring.
  - Blinded prompt: the model is never told which AI produced the response.
  - JSON response enforcement: Groq's response_format=json_object ensures
    parseable output; one retry is attempted on parse failure.
  - Exponential back-off on 429 rate-limit errors.
  - Checkpoint every CHECKPOINT_EVERY questions — safe to interrupt and resume.

Usage:
    python scripts/groq_judge.py
    python scripts/groq_judge.py --dry-run          # preview first 3 prompts
    python scripts/groq_judge.py --layers 1,3       # judge specific layers only
    python scripts/groq_judge.py --resume data/eval/comparison/judged_results_<ts>.json

Output:
    data/eval/comparison/judged_results_<timestamp>.json   — per-question detail
    data/eval/comparison/judged_summary_<timestamp>.json   — aggregate stats

Environment variables (loaded from .env):
    GROQ_API_KEY   — required
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

# ── load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────
JUDGE_MODEL: str        = "llama-3.1-8b-instant"
CHECKPOINT_EVERY: int   = 10
MAX_RETRIES: int        = 6        # retries on parse failure or transient error
BACKOFF_BASE: float     = 15.0     # seconds; doubles on each rate-limit hit
MAX_BACKOFF: float      = 120.0
INTER_CALL_DELAY: float = 4.0      # seconds to wait between every API call (TPM pacing)

# Layer rubrics — injected verbatim into the judge prompt
RUBRICS: dict[int, str] = {
    1: "2 = correct answer with sound clinical reasoning  |  1 = partially correct or weak reasoning  |  0 = wrong or hallucinated",
    3: "2 = correct answer with sound clinical reasoning  |  1 = partially correct or weak reasoning  |  0 = wrong or hallucinated",
    4: "3 = complete (correct diagnosis AND full management plan)  |  2 = mostly correct, minor omissions  |  1 = partial, significant gaps  |  0 = wrong or hallucinated",
    5: "2 = correctly reflects the ASH guideline with sound reasoning  |  1 = partially correct or imprecise  |  0 = wrong or contradicts the guideline",
}

MAX_SCORES: dict[int, int] = {1: 2, 2: 1, 3: 2, 4: 3, 5: 2}

# ── file paths ────────────────────────────────────────────────────────────────
EVAL_FILES: dict[int, Path] = {
    1: Path("data/eval/layer1_custom_notes.json"),
    2: Path("data/eval/layer2_benchmark.json"),
    3: Path("data/eval/layer3_combined.json"),
    4: Path("data/eval/layer4_clinical_cases.json"),
    5: Path("data/eval/layer5_ASH.json"),
}
PHI3_ANSWERS_FILE: Path = Path("data/eval/phi3_mini_answers.json")
CLAUDE_FILE: Path        = Path("data/eval/SCD_Answer_Key_ALL.json")
OUTPUT_DIR: Path         = Path("data/eval/comparison")


# ── data loading ──────────────────────────────────────────────────────────────
def load_json(path: Path, label: str) -> list | dict:
    """
    Load and parse a JSON file, exiting with a clear error on failure.

    Args:
        path:  File path to read.
        label: Human-readable name for error messages.

    Returns:
        Parsed JSON structure (list or dict).

    Raises:
        SystemExit: On missing file or JSON parse error.
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


def load_gold_standard() -> dict[str, dict]:
    """
    Load all gold-standard answers from the layer eval files.

    Layer 1 may be wrapped in a ``{"fullContent": [...]}`` dict; all other
    layers are plain arrays.

    Returns:
        Dict mapping question id → record dict with keys:
        id, layer, category, source, question, case, rationale, gold_answer, max_score.
    """
    gold: dict[str, dict] = {}
    for layer, path in EVAL_FILES.items():
        if not path.exists():
            print(f"  [!] Gold file for Layer {layer} not found: {path}")
            continue
        raw = load_json(path, f"Layer {layer} eval file")
        records = raw["fullContent"] if isinstance(raw, dict) and "fullContent" in raw else raw
        for r in records:
            rid = r.get("id")
            if not rid:
                continue
            gold[rid] = {
                "id":          rid,
                "layer":       layer,
                "category":    r.get("category", ""),
                "source":      r.get("source", ""),
                "question":    r.get("question", ""),
                "case":        r.get("case", ""),
                "rationale":   r.get("rationale", ""),
                "gold_answer": r.get("answer", ""),
                "max_score":   MAX_SCORES[layer],
            }
    return gold


def load_phi3_responses() -> dict[str, str]:
    """
    Load phi3:mini responses from the consolidated answers file.

    Returns:
        Dict mapping question id → model_response string.
    """
    records = load_json(PHI3_ANSWERS_FILE, "phi3:mini answers file")
    return {r["id"]: r.get("model_response", "") for r in records if "id" in r}


def load_claude_responses() -> dict[str, str]:
    """
    Load Claude Opus 4.8 responses from SCD_Answer_Key_ALL.json.

    Returns:
        Dict mapping question id → answer string.
    """
    records = load_json(CLAUDE_FILE, "Claude answers file")
    return {r["id"]: r.get("answer", "") for r in records if "id" in r}


# ── question alignment ────────────────────────────────────────────────────────
def align_questions(
    gold: dict[str, dict],
    phi3: dict[str, str],
    claude: dict[str, str],
    layers: list[int],
) -> list[dict]:
    """
    Combine gold, phi3, and Claude data into one aligned list per question.

    Missing model responses are recorded as ``"[NO RESPONSE]"`` rather than
    silently omitted, so coverage gaps are visible in the output.

    Args:
        gold:   Gold-standard records keyed by question id.
        phi3:   phi3:mini responses keyed by question id.
        claude: Claude responses keyed by question id.
        layers: Layer numbers to include.

    Returns:
        List of aligned question dicts, sorted by layer then id.
    """
    aligned = []
    for rid, g in gold.items():
        if g["layer"] not in layers:
            continue
        aligned.append({
            "id":              rid,
            "layer":           g["layer"],
            "category":        g["category"],
            "source":          g["source"],
            "question":        g["question"],
            "case":            g.get("case", ""),
            "rationale":       g.get("rationale", ""),
            "gold_answer":     g["gold_answer"],
            "max_score":       g["max_score"],
            "phi3_response":   phi3.get(rid, "[NO RESPONSE]"),
            "claude_response": claude.get(rid, "[NO RESPONSE]"),
            "phi3_score":      None,
            "phi3_rationale":  None,
            "claude_score":    None,
            "claude_rationale": None,
            "scoring_method":  None,
            "notes":           "",
        })
    aligned.sort(key=lambda x: (x["layer"], x["id"]))
    return aligned


# ── Layer 2 keyword auto-scoring ──────────────────────────────────────────────
def keyword_match(gold_answer: str, model_response: str) -> int:
    """
    Auto-score a Layer 2 MCQ response by substring match on the gold answer.

    The gold answer is normalised (lowercase, stripped punctuation) before
    matching so short phrases like ``"P. falciparum"`` or ``"100%"`` are
    reliably detected within longer free-text responses.

    Args:
        gold_answer:    The correct answer string.
        model_response: The model's response text.

    Returns:
        1 if gold answer is found in the response, 0 otherwise.
    """
    if not gold_answer or model_response == "[NO RESPONSE]":
        return 0
    gold_norm     = gold_answer.strip().lower().strip("*\"'`")
    response_norm = model_response.lower()
    if gold_norm in response_norm:
        return 1
    core = re.sub(r"[^a-z0-9%.\-]", " ", gold_norm).strip().split()
    if core and any(word in response_norm for word in core if len(word) > 3):
        return 1
    return 0


def auto_score_layer2(questions: list[dict]) -> None:
    """
    Apply keyword-match scoring to all Layer 2 entries in-place.

    Args:
        questions: Full aligned question list (mutated in-place).
    """
    for q in questions:
        if q["layer"] != 2:
            continue
        q["phi3_score"]     = keyword_match(q["gold_answer"], q["phi3_response"])
        q["claude_score"]   = keyword_match(q["gold_answer"], q["claude_response"])
        q["phi3_rationale"]   = "auto_keyword_match"
        q["claude_rationale"] = "auto_keyword_match"
        q["scoring_method"] = "auto_keyword"


# ── judge prompt ──────────────────────────────────────────────────────────────
def build_judge_prompt(q: dict, model_response: str) -> str:
    """
    Build the blinded scoring prompt for the Groq judge.

    The prompt deliberately omits any reference to which model produced the
    response. Layer 5 ASH rationale is included as evaluator context when
    available. Layer 4 clinical case context is prepended to the question.

    Args:
        q:              Aligned question dict.
        model_response: The specific model response to score (phi3 or Claude).

    Returns:
        The complete prompt string ready for the Groq API.
    """
    layer   = q["layer"]
    rubric  = RUBRICS[layer]
    max_s   = q["max_score"]

    # Build question block — prepend clinical case for Layer 4
    if q.get("case"):
        question_block = f"CLINICAL CASE:\n{q['case']}\n\nQUESTION:\n{q['question']}"
    else:
        question_block = f"QUESTION:\n{q['question']}"

    # Include ASH rationale for Layer 5 as evaluator context
    rationale_block = ""
    if layer == 5 and q.get("rationale"):
        rationale_block = f"\nEVALUATOR CONTEXT (ASH rationale — do NOT reveal to candidate):\n{q['rationale']}\n"

    return f"""You are an expert medical examiner assessing answers to sickle cell disease questions.
Your task is to score the CANDIDATE RESPONSE against the GOLD STANDARD ANSWER using the RUBRIC.

{question_block}

GOLD STANDARD ANSWER:
{q['gold_answer']}
{rationale_block}
CANDIDATE RESPONSE:
{model_response}

RUBRIC (maximum score: {max_s}):
{rubric}

Instructions:
- Score ONLY based on medical accuracy relative to the gold standard answer.
- Do NOT guess or consider which AI system produced this response.
- Do NOT penalise for style, length, or format — only correctness matters.
- If the response is "[NO RESPONSE]", the score must be 0.

Respond ONLY with a valid JSON object in this exact format — no extra text:
{{"score": <integer 0 to {max_s}>, "rationale": "<one concise sentence explaining the score>"}}"""


# ── Groq API call ─────────────────────────────────────────────────────────────
def call_groq_judge(
    client: Groq,
    prompt: str,
    question_id: str,
    model_label: str,
) -> tuple[int | None, str]:
    """
    Send a scoring prompt to the Groq judge and parse the JSON response.

    Retries up to MAX_RETRIES times on JSON parse failure. Applies
    exponential back-off on 429 rate-limit errors. On persistent failure
    returns (None, error_message) so the run continues rather than crashing.

    Args:
        client:       Authenticated Groq client instance.
        prompt:       The full judge prompt string.
        question_id:  Used in log messages only.
        model_label:  "phi3" or "claude" — used in log messages.

    Returns:
        Tuple of (score: int | None, rationale: str).
    """
    backoff = BACKOFF_BASE

    # Pace every call to stay within Groq free-tier TPM limits
    time.sleep(INTER_CALL_DELAY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=256,
            )
            raw_text = response.choices[0].message.content.strip()

            parsed = json.loads(raw_text)
            score     = parsed.get("score")
            rationale = parsed.get("rationale", "")

            if not isinstance(score, int):
                raise ValueError(f"Non-integer score: {score!r}")

            return score, rationale

        except RateLimitError:
            wait = min(backoff, MAX_BACKOFF)
            print(
                f"\n  [RATE LIMIT] {question_id}/{model_label} — "
                f"waiting {wait:.0f}s before retry {attempt}/{MAX_RETRIES}...",
                end=" ", flush=True,
            )
            time.sleep(wait)
            backoff *= 2
            continue

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            if attempt < MAX_RETRIES:
                print(
                    f"\n  [PARSE ERROR attempt {attempt}] {question_id}/{model_label}: {exc} — retrying",
                    file=sys.stderr,
                )
                time.sleep(1)
                continue
            print(
                f"\n  [PARSE FAIL] {question_id}/{model_label} after {MAX_RETRIES} attempts: {exc}",
                file=sys.stderr,
            )
            return None, f"parse_error: {exc}"

        except APIError as exc:
            print(
                f"\n  [API ERROR] {question_id}/{model_label}: {exc}",
                file=sys.stderr,
            )
            return None, f"api_error: {exc}"

    return None, "max_retries_exceeded"


# ── persistence ───────────────────────────────────────────────────────────────
def save_json_file(data: list | dict, path: Path) -> None:
    """
    Write data to a JSON file, creating parent directories as needed.

    Args:
        data: Serialisable object to write.
        path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def save_checkpoint(questions: list[dict], timestamp: str, count: int) -> None:
    """
    Save an intermediate checkpoint of the full question list.

    Args:
        questions: Full aligned list (all entries, including unscored ones).
        timestamp: Run timestamp string (used in the filename).
        count:     Number of questions judged so far this session.
    """
    path = OUTPUT_DIR / f"judged_results_{timestamp}_checkpoint_{count}.json"
    save_json_file(questions, path)
    print(f"\n  [checkpoint] {count} judged — saved to {path.name}\n")


# ── summary ───────────────────────────────────────────────────────────────────
def build_summary(questions: list[dict]) -> dict:
    """
    Compute aggregate scoring statistics from the complete question list.

    Only entries where both scores are non-None are included in percentages.
    Breakdowns are provided by layer and by category.

    Args:
        questions: Full aligned and scored question list.

    Returns:
        Summary dict with overall, by_layer, and by_category breakdowns.
    """
    def pct(score: int, max_s: int) -> float:
        """Return score as a percentage of max_score."""
        return (score / max_s * 100) if max_s > 0 else 0.0

    scored = [
        q for q in questions
        if q["phi3_score"] is not None and q["claude_score"] is not None
    ]

    def aggregate(entries: list[dict]) -> dict:
        """Return aggregate stats for a group of scored entries."""
        if not entries:
            return {"count": 0, "phi3_pct": None, "claude_pct": None, "winner": None}
        phi3_pcts   = [pct(q["phi3_score"],   q["max_score"]) for q in entries]
        claude_pcts = [pct(q["claude_score"], q["max_score"]) for q in entries]
        phi3_avg    = sum(phi3_pcts)   / len(phi3_pcts)
        claude_avg  = sum(claude_pcts) / len(claude_pcts)
        winner = (
            "phi3:mini"        if phi3_avg > claude_avg else
            "claude_opus_4.8"  if claude_avg > phi3_avg else
            "tie"
        )
        return {
            "count":      len(entries),
            "phi3_pct":   round(phi3_avg,   1),
            "claude_pct": round(claude_avg, 1),
            "winner":     winner,
        }

    by_layer = {}
    for layer in sorted({q["layer"] for q in questions}):
        by_layer[f"layer{layer}"] = aggregate(
            [q for q in scored if q["layer"] == layer]
        )

    by_cat = {}
    for cat in sorted({q["category"] for q in questions}):
        by_cat[cat] = aggregate([q for q in scored if q["category"] == cat])

    l2_auto  = len([q for q in scored if q["scoring_method"] == "auto_keyword"])
    l_judged = len([q for q in scored if q["scoring_method"] == "groq_llm_judge"])

    return {
        "generated_at":     datetime.now().isoformat(),
        "judge_model":      JUDGE_MODEL,
        "total_questions":  len(questions),
        "scored":           len(scored),
        "unscored":         len(questions) - len(scored),
        "auto_scored_l2":   l2_auto,
        "llm_judged":       l_judged,
        "overall":          aggregate(scored),
        "by_layer":         by_layer,
        "by_category":      by_cat,
    }


def print_summary(summary: dict) -> None:
    """
    Print a formatted comparison summary to stdout.

    Args:
        summary: Summary dict produced by build_summary().
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  GROQ JUDGE SUMMARY  —  phi3:mini  vs  Claude Opus 4.8")
    print(f"  Judge model: {summary['judge_model']}")
    print(sep)
    print(f"  Total questions : {summary['total_questions']}")
    print(f"  Scored          : {summary['scored']}  "
          f"(auto L2: {summary['auto_scored_l2']} | LLM judged: {summary['llm_judged']})")
    print(f"  Unscored        : {summary['unscored']}")
    o = summary["overall"]
    if o["count"]:
        print(f"\n  OVERALL   phi3 {o['phi3_pct']:.1f}%  |  Claude {o['claude_pct']:.1f}%  "
              f"|  Winner: {o['winner']}")
    print("\n  By layer:")
    for key, stats in summary["by_layer"].items():
        if stats["count"]:
            print(f"    {key:<10}  phi3 {stats['phi3_pct']:>5.1f}%  |  "
                  f"Claude {stats['claude_pct']:>5.1f}%  |  {stats['winner']}")
    print("\n  By category:")
    for cat, stats in summary["by_category"].items():
        if stats["count"]:
            print(f"    {cat:<35}  phi3 {stats['phi3_pct']:>5.1f}%  |  "
                  f"Claude {stats['claude_pct']:>5.1f}%")
    print(sep + "\n")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Entry point: parse args, load data, judge all non-L2 questions, save output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score phi3:mini and Claude Opus 4.8 responses using the Groq "
            f"LLM-as-Judge ({JUDGE_MODEL}). Layer 2 MCQ is auto-scored "
            "via keyword matching; all other layers are sent to the judge."
        )
    )
    parser.add_argument(
        "--layers",
        default="1,2,3,4,5",
        help="Comma-separated layer numbers to include (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first 3 judge prompts (one model each) and exit — no API calls made.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a previous judged_results_*.json file to resume from.",
    )
    args = parser.parse_args()

    layers     = [int(x.strip()) for x in args.layers.split(",") if x.strip().isdigit()]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")

    # ── validate API key ──────────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "[ERROR] GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=gsk_...",
            file=sys.stderr,
        )
        sys.exit(1)

    client = Groq(api_key=api_key)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\n  Judge model    : {JUDGE_MODEL}")
    print(f"  Layers         : {layers}")
    print(f"  Output dir     : {OUTPUT_DIR.resolve()}\n")

    print("  Loading gold standard answers...")
    gold = load_gold_standard()
    print(f"    {len(gold)} gold answers loaded")

    print("  Loading phi3:mini responses...")
    phi3 = load_phi3_responses()
    print(f"    {len(phi3)} phi3 responses loaded")

    print("  Loading Claude Opus 4.8 responses...")
    claude = load_claude_responses()
    print(f"    {len(claude)} Claude responses loaded\n")

    # ── align ─────────────────────────────────────────────────────────────────
    questions = align_questions(gold, phi3, claude, layers)
    print(f"  Aligned {len(questions)} questions across layers {layers}")

    # ── resume ────────────────────────────────────────────────────────────────
    already_scored: set[str] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            prev = load_json(resume_path, "Resume file")
            prev_map = {r["id"]: r for r in prev}
            for q in questions:
                if q["id"] in prev_map and prev_map[q["id"]].get("scoring_method"):
                    q.update(prev_map[q["id"]])
                    # Only mark as already scored if we actually got integer scores
                    if q.get("phi3_score") is not None and q.get("claude_score") is not None:
                        already_scored.add(q["id"])
            print(f"  Resumed: {len(already_scored)} already scored, "
                  f"{len(questions) - len(already_scored)} remaining")
        else:
            print(f"  [!] Resume file not found: {resume_path}")

    # ── Layer 2 auto-scoring ──────────────────────────────────────────────────
    if 2 in layers:
        l2_pending = [q for q in questions if q["layer"] == 2 and q["id"] not in already_scored]
        if l2_pending:
            print(f"\n  Auto-scoring {len(l2_pending)} Layer 2 MCQ questions (keyword match)...")
            auto_score_layer2(questions)
            for q in questions:
                if q["layer"] == 2:
                    already_scored.add(q["id"])
            print("    Done.")

    # ── dry-run: print 3 sample prompts and exit ──────────────────────────────
    if args.dry_run:
        non_l2 = [q for q in questions if q["layer"] != 2][:3]
        print(f"\n{'='*72}")
        print("  DRY RUN — first 3 judge prompts (phi3 response shown)")
        print(f"{'='*72}\n")
        for i, q in enumerate(non_l2, 1):
            prompt = build_judge_prompt(q, q["phi3_response"])
            print(f"--- Prompt {i} ({q['id']} / Layer {q['layer']}) ---")
            print(prompt.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))
            print()
        print("  [Dry run complete — no API calls were made]\n")
        sys.exit(0)

    # ── LLM judging for Layers 1, 3, 4, 5 ────────────────────────────────────
    to_judge = [q for q in questions if q["layer"] != 2 and q["id"] not in already_scored]
    # Each question requires 2 API calls (one per model)
    total_calls = len(to_judge) * 2
    print(f"\n  {len(to_judge)} questions to judge via Groq ({total_calls} API calls).\n")

    judged_count = 0

    for i, q in enumerate(to_judge, start=1):
        qid = q["id"]

        # ── Score phi3:mini ──────────────────────────────────────────────────
        print(f"  [{i}/{len(to_judge)}] {qid} (L{q['layer']}) — judging phi3...", end=" ", flush=True)
        if q["phi3_response"] == "[NO RESPONSE]":
            q["phi3_score"]     = 0
            q["phi3_rationale"] = "no_response"
            print("(no response → 0)", end=" ", flush=True)
        else:
            phi3_prompt = build_judge_prompt(q, q["phi3_response"])
            score, rationale = call_groq_judge(client, phi3_prompt, qid, "phi3")
            q["phi3_score"]     = score
            q["phi3_rationale"] = rationale
            print(f"score={score}", end="  ", flush=True)

        # ── Score Claude ─────────────────────────────────────────────────────
        print("judging claude...", end=" ", flush=True)
        if q["claude_response"] == "[NO RESPONSE]":
            q["claude_score"]     = 0
            q["claude_rationale"] = "no_response"
            print("(no response → 0)")
        else:
            claude_prompt = build_judge_prompt(q, q["claude_response"])
            score, rationale = call_groq_judge(client, claude_prompt, qid, "claude")
            q["claude_score"]     = score
            q["claude_rationale"] = rationale
            print(f"score={score}")

        q["scoring_method"] = "groq_llm_judge"
        q["judge_model"]    = JUDGE_MODEL
        already_scored.add(qid)
        judged_count += 1

        # ── checkpoint ───────────────────────────────────────────────────────
        if judged_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(questions, timestamp, judged_count)

    # ── final save ────────────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / f"judged_results_{timestamp}.json"
    summary_path = OUTPUT_DIR / f"judged_summary_{timestamp}.json"

    try:
        save_json_file(questions, results_path)
        summary = build_summary(questions)
        save_json_file(summary, summary_path)
    except OSError as exc:
        print(f"\n[ERROR] Failed to write output: {exc}", file=sys.stderr)
        sys.exit(1)

    print_summary(summary)
    print(f"  Results -> {results_path.resolve()}")
    print(f"  Summary -> {summary_path.resolve()}\n")


if __name__ == "__main__":
    main()
