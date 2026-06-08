import json
import ollama
import os
from datetime import datetime

# ── file map ────────────────────────────────────────────────────────────────
EVAL_FILES = {
    1: "data/eval/layer1_custom_notes.json",
    2: "data/eval/layer2_benchmark.json",
    3: "data/eval/layer3_combined.json",
    4: "data/eval/layer4_clinical_cases.json",
}

SYSTEM_PROMPT = (
    "You are a medical AI assistant specialised in sickle cell disease. "
    "Answer clinical questions accurately and concisely. "
    "If you are uncertain, say so clearly rather than guessing."
)

# ── loaders ─────────────────────────────────────────────────────────────────
def load_all_questions():
    all_questions = []
    for layer, path in EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"  [!] Layer {layer} file not found: {path} — skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # FIX 1: Handle Layer 1 nested wrapper vs normal arrays
        if isinstance(data, dict) and "fullContent" in data:
            questions = data["fullContent"]
        elif isinstance(data, list):
            questions = data
        else:
            print(f"  [!] Unknown format in Layer {layer} — skipping")
            continue

        # attach layer number to every entry
        for q in questions:
            q["layer"] = layer
            
        all_questions.extend(questions)
        print(f"  [+] Layer {layer}: {len(questions)} questions loaded")
    return all_questions


def build_prompt(q):
    """
    Construct the prompt sent to the model.
    Layer 4 has a 'case' field that provides clinical context —
    prepend it to the question so the model has the full picture.
    """
    if "case" in q and q["case"]:
        return f"Clinical case:\n{q['case']}\n\nQuestion: {q['question']}"
    return q["question"]


# ── model call ───────────────────────────────────────────────────────────────
def ask_model(model_name, prompt):
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
    )
    return response["message"]["content"]


# ── scoring helper ───────────────────────────────────────────────────────────
def score_entry(q, response):
    """
    Print gold answer (and rubric for layer 4) so you can score immediately.
    Returns the scored result dict — you type the score at the prompt.
    """
    print("\n" + "─" * 70)
    print(f"ID       : {q['id']}")
    print(f"Layer    : {q['layer']}  |  Category: {q.get('category', 'n/a')}")
    if "case" in q and q["case"]:
        print(f"\nCASE:\n{q['case']}")
    print(f"\nQUESTION:\n{q['question']}")
    print(f"\nMODEL RESPONSE:\n{response}")
    print(f"\nGOLD ANSWER:\n{q['answer']}")

    # layer 2 multiple choice — exact match
    if q["layer"] == 2:
        print("\n[Layer 2 — multiple choice: 1 = correct, 0 = wrong]")
        score = input("Score (1/0): ").strip()
        max_score = 1

    # layer 4 — rubric scoring
    elif q["layer"] == 4 and "rubric" in q:
        print("\nRUBRIC:")
        for k, v in q["rubric"].items():
            print(f"  {k} — {v}")
        score = input("Score (0/1/2/3): ").strip()
        max_score = 3

    # layers 1 and 3 — standard rubric
    else:
        print("\nSCORE GUIDE: 2=correct+reasoning  1=partial  0=wrong/hallucinated")
        score = input("Score (0/1/2): ").strip()
        max_score = 2

    notes = input("Notes (optional, press enter to skip): ").strip()

    return {
        "id":            q["id"],
        "layer":         q["layer"],
        "category":      q.get("category", ""),
        "source":        q.get("source", ""),
        "question":      q["question"],
        "case":          q.get("case", ""),
        "gold_answer":   q["answer"],
        "model_response": response,
        "score":         int(score) if score.isdigit() else None,
        "max_score":     max_score,
        "notes":         notes,
    }


# ── main runner ──────────────────────────────────────────────────────────────
def run_baseline(model_name, questions, output_dir="data/eval/baselines",
                 start_from=0, auto_score=False):
    """
    Run the full baseline eval for one model.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # FIX 2b: If resuming, try to load existing run so history isn't severed in final score computation
    # Alternatively, ensure results tracking is aware of index values.
    # To keep your checkpoint arrays structurally complete, we retain slots for missed metrics if running fresh.

    print(f"\n{'='*70}")
    print(f"Model    : {model_name}")
    print(f"Questions: {len(questions)}  (starting from index {start_from})")
    print(f"Scoring  : {'batch later' if auto_score else 'interactive (score as you go)'}")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions):
        if i < start_from:
            continue  # Fast forward to your chosen entry point without breaking internal 'i' counts

        prompt = build_prompt(q)
        print(f"\n[{i+1}/{len(questions)}] {q['id']} — asking model...", end=" ", flush=True)

        try:
            response = ask_model(model_name, prompt)
            print("done")
        except Exception as e:
            print(f"\n  [!] Error: {e} — skipping")
            continue

        if auto_score:
            result = {
                "id":             q["id"],
                "layer":          q["layer"],
                "category":       q.get("category", ""),
                "source":         q.get("source", ""),
                "question":       q["question"],
                "case":           q.get("case", ""),
                "gold_answer":    q["answer"],
                "model_response": response,
                "score":          None,
                "max_score":      3 if q["layer"] == 4 else (1 if q["layer"] == 2 else 2),
                "notes":          "",
            }
        else:
            result = score_entry(q, response)
            
        results.append(result)

        # checkpoint save every 10 processed questions
        if len(results) % 10 == 0:
            _save(results, model_name, output_dir, suffix=f"checkpoint_idx_{i+1}")
            print(f"\n  [checkpoint saved — {len(results)} questions completed this session]\n")

    # final save
    saved_path = _save(results, model_name, output_dir, suffix="final")
    print(f"\n{'='*70}")
    print(f"Done. Results saved to: {saved_path}")
    _print_summary(results)
    return results


def _save(results, model_name, output_dir, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_model = model_name.replace(":", "_").replace("/", "_")
    filename = f"{safe_model}_{timestamp}_{suffix}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def _print_summary(results):
    scored = [r for r in results if r["score"] is not None]
    if not scored:
        print("No scores recorded yet.")
        return

    print(f"\nSUMMARY")
    print(f"{'─'*40}")
    print(f"Scored     : {len(scored)} / {len(results)}")

    # overall percentage (normalise each score to 0-1 range)
    pct_scores = [r["score"] / r["max_score"] for r in scored]
    overall = sum(pct_scores) / len(pct_scores) * 100
    print(f"Overall    : {overall:.1f}%")

    # by layer
    print("\nBy layer:")
    for layer in sorted(set(r["layer"] for r in scored)):
        layer_results = [r for r in scored if r["layer"] == layer]
        layer_pct = sum(r["score"] / r["max_score"] for r in layer_results) / len(layer_results) * 100
        print(f"  Layer {layer}: {layer_pct:.1f}%  ({len(layer_results)} questions)")

    # by category
    print("\nBy category:")
    cats = sorted(set(r["category"] for r in scored if r["category"]))
    for cat in cats:
        cat_results = [r for r in scored if r["category"] == cat]
        cat_pct = sum(r["score"] / r["max_score"] for r in cat_results) / len(cat_results) * 100
        print(f"  {cat:<30} {cat_pct:.1f}%  ({len(cat_results)} questions)")


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading eval set...")
    questions = load_all_questions()
    print(f"\nTotal: {len(questions)} questions across {len(EVAL_FILES)} layers\n")

    print("Which model do you want to run?")
    print("  1. phi3:mini  (fast on CPU, ~3B)")
    print("  2. mistral    (slower on CPU, ~7B)")
    print("  3. custom     (type your own Ollama model name)")
    choice = input("\nChoice (1/2/3): ").strip()

    model_map = {"1": "phi3:mini", "2": "mistral"}
    if choice in model_map:
        model_name = model_map[choice]
    else:
        model_name = input("Enter model name: ").strip()

    print("\nScoring mode:")
    print("  1. Score as I go (interactive) — recommended for first run")
    print("  2. Save all responses first, score later (faster)")
    mode = input("Choice (1/2): ").strip()
    auto_score = mode == "2"

    resume = input("\nResume from question index? (press enter for 0): ").strip()
    start_from = int(resume) if resume.isdigit() else 0

    run_baseline(model_name, questions,
                 start_from=start_from,
                 auto_score=auto_score)