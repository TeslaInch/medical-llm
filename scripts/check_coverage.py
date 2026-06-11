"""check_coverage.py — diagnostic: compare questions vs phi3.5:mini answers."""
import json
from collections import Counter
from pathlib import Path

# ── load sources ──────────────────────────────────────────────────────────────
with open("data/eval/all_questions.json", "r", encoding="utf-8") as f:
    all_questions = json.load(f)

with open("data/eval/phi3.5_mini_answers.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

with open("data/eval/SCD_Answer_Key_ALL.json", "r", encoding="utf-8") as f:
    answer_key = json.load(f)

# ── id sets ───────────────────────────────────────────────────────────────────
all_q_ids   = {q["id"] for q in all_questions}
answered_ids = {a["id"] for a in answers}
ak_ids       = {r["id"] for r in answer_key}

missing_from_questions = all_q_ids - answered_ids
missing_from_ak        = ak_ids - answered_ids

print("=" * 60)
print("  PHI3:MINI COVERAGE REPORT")
print("=" * 60)
print(f"  all_questions.json total     : {len(all_q_ids)}")
print(f"  SCD_Answer_Key_ALL.json total: {len(ak_ids)}")
print(f"  phi3.5:mini answers extracted  : {len(answered_ids)}")
print()
print(f"  Missing vs all_questions.json: {len(missing_from_questions)}")
print(f"  Missing vs answer key        : {len(missing_from_ak)}")
print("=" * 60)

# ── layer breakdown of missing (vs all_questions) ─────────────────────────────
missing_records = [q for q in all_questions if q["id"] in missing_from_questions]
layer_counts = Counter(q.get("layer") for q in missing_records)
print("\n  Missing by layer (vs all_questions.json):")
for layer, count in sorted(layer_counts.items()):
    print(f"    Layer {layer}: {count} missing")

# ── layer breakdown of missing (vs answer key) ────────────────────────────────
ak_missing_records = [r for r in answer_key if r["id"] in missing_from_ak]
ak_layer_counts = Counter(r.get("layer") for r in ak_missing_records)
print("\n  Missing by layer (vs SCD_Answer_Key_ALL.json):")
for layer, count in sorted(ak_layer_counts.items()):
    print(f"    Layer {layer}: {count} missing")

# ── sample of missing ids ─────────────────────────────────────────────────────
sample = sorted(missing_from_ak)[:20]
print("\n  Sample of missing IDs (first 20):")
for qid in sample:
    print(f"    {qid}")

print("=" * 60)
