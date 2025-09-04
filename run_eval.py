# eval.py
# Runs your pipeline (interpret → plan → retrieve → answer) for each test case
# in tests.yaml and writes eval_results.json with raw answers + metadata.

import json
import argparse
import traceback
from pathlib import Path

# Optional dependency: PyYAML
try:
    import yaml
except ImportError:
    raise SystemExit("Please install PyYAML: pip install pyyaml")

# Import your agent pipeline
from agent import interpret_then_plan, retrieve, answer  # noqa: E402


def _uniq_urls_from_evidence(evidence_table):
    seen, out = set(), []
    for row in evidence_table or []:
        u = (row.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def run_one(question: str):
    """
    Runs one full query through the agent.
    Returns a dict:
      {
        "question": ...,
        "interpret": {...},
        "plan": {...},
        "evidence_table": [...],
        "answer_text": "...",
        "urls": [...],
        "error": None | "..."
      }
    """
    rec = {
        "question": question,
        "interpret": None,
        "plan": None,
        "evidence_table": [],
        "answer_text": "",
        "urls": [],
        "error": None,
    }

    try:
        pipe = interpret_then_plan(question, history=[])
        interp = pipe.get("interpret", {}) or {}
        plan = pipe.get("plan", {}) or {}
        rec["interpret"] = interp
        rec["plan"] = plan

        # Non-political? Let answer() handle the gentle redirect; no retrieval.
        if (interp.get("scope") or "").strip().lower() == "non-political":
            ans = answer(interp, [])
            rec["answer_text"] = ans
            rec["urls"] = []
            return rec

        # Otherwise retrieve → answer
        ev = retrieve(interp, plan) or {}
        table = ev.get("evidence_table", []) or []
        rec["evidence_table"] = table
        rec["urls"] = _uniq_urls_from_evidence(table)
        ans = answer(interp, table)
        rec["answer_text"] = ans
        return rec

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        return rec


def main():
    ap = argparse.ArgumentParser(description="Run civic-llm-agent evals")
    ap.add_argument("--tests", default="tests.yaml", help="Path to tests.yaml")
    ap.add_argument("--out", default="eval_results.json", help="Where to write results JSON")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N tests (0 = all)")
    args = ap.parse_args()

    tests_path = Path(args.tests)
    if not tests_path.exists():
        raise SystemExit(f"Missing tests file: {tests_path}")

    with tests_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tests = data.get("tests", [])
    if args.limit and args.limit > 0:
        tests = tests[: args.limit]

    results = []
    for t in tests:
        q = t.get("question", "").strip()
        tid = t.get("id", "")
        if not q:
            results.append({
                "id": tid, "question": q, "error": "empty question"
            })
            continue

        rec = run_one(q)
        rec["id"] = tid
        rec["requires_citations"] = bool(t.get("requires_citations", False))
        rec["expect_uncertainty"] = bool(t.get("expect_uncertainty", False))
        rec["expect_refusal"] = bool(t.get("expect_refusal", False))
        results.append(rec)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results → {out_path}")


if __name__ == "__main__":
    main()
