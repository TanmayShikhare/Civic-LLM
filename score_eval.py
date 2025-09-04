# score_eval.py
# Scores eval_results.json (from eval.py) against tests.yaml using simple,
# transparent heuristics (citations, uncertainty cue, refusal cue, etc).

import re
import json
import argparse
from pathlib import Path

# Optional dependency: PyYAML
try:
    import yaml
except ImportError:
    raise SystemExit("Please install PyYAML: pip install pyyaml")


# ---------- Heuristics ----------
_CITATION_RE = re.compile(r"\[\s*\d+\s*\]")
_REFUSAL_CUES = [
    "this chatbot is focused on politics",
    "this chatbot focuses on politics",
    "not political",
    "out of scope",
    "civic topics",
    "rephrase your question",
]
_UNCERTAINTY_CUES = [
    "evidence is limited",
    "mixed evidence",
    "uncertain",
    "may change",
    "ongoing",
    "limitations",
    "as of ",
    "note limits",
    "confidence is lower",
]

def has_citations(text: str, min_count: int = 1) -> bool:
    if not text:
        return False
    matches = _CITATION_RE.findall(text)
    return len(matches) >= min_count

def has_uncertainty(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in _UNCERTAINTY_CUES)

def is_refusal(rec: dict) -> bool:
    """
    Passes if either:
      - interpret scope is "non-political"
      - OR final answer has a refusal cue
    """
    interp = rec.get("interpret") or {}
    scope = (interp.get("scope") or "").strip().lower()
    if scope == "non-political":
        return True
    text = (rec.get("answer_text") or "").lower()
    return any(k in text for k in _REFUSAL_CUES)

def is_non_refusal(text: str) -> bool:
    """Ensure answer doesn't look like a refusal when not expected."""
    if not text:
        return False
    low = text.lower()
    return not any(k in low for k in _REFUSAL_CUES)


def main():
    ap = argparse.ArgumentParser(description="Score civic-llm-agent eval results")
    ap.add_argument("--tests", default="tests.yaml", help="Path to tests.yaml")
    ap.add_argument("--results", default="eval_results.json", help="Path to eval_results.json")
    ap.add_argument("--min_citations", type=int, default=1, help="Minimum citation marks [#] required")
    args = ap.parse_args()

    tests_path = Path(args.tests)
    res_path = Path(args.results)

    if not tests_path.exists():
        raise SystemExit(f"Missing tests file: {tests_path}")
    if not res_path.exists():
        raise SystemExit(f"Missing results file: {res_path} (run: python eval.py)")

    with tests_path.open("r", encoding="utf-8") as f:
        tests_yaml = yaml.safe_load(f) or {}
    tests = {t.get("id"): t for t in (tests_yaml.get("tests", []) or [])}

    results = json.loads(res_path.read_text(encoding="utf-8"))

    total = 0
    passed = 0
    rows = []

    for rec in results:
        tid = rec.get("id")
        spec = tests.get(tid, {})
        q = rec.get("question", "")
        ans = rec.get("answer_text") or ""
        err = rec.get("error")

        # default verdict object
        verdict = {
            "id": tid,
            "question": q,
            "error": err,
            "checks": {},
            "pass": True,
        }

        # If pipeline crashed → fail test loudly
        if err:
            verdict["checks"]["runtime_error"] = False
            verdict["pass"] = False
            rows.append(verdict)
            total += 1
            continue
        else:
            verdict["checks"]["runtime_error"] = True

        # --- requires_citations?
        if spec.get("requires_citations", False):
            ok = has_citations(ans, min_count=args.min_citations)
            verdict["checks"]["citations"] = ok
            verdict["pass"] = verdict["pass"] and ok

        # --- expect_uncertainty?
        if spec.get("expect_uncertainty", False):
            ok = has_uncertainty(ans)
            verdict["checks"]["uncertainty"] = ok
            verdict["pass"] = verdict["pass"] and ok
        else:
            # If not expected, we don't punish having uncertainty language.
            verdict["checks"]["uncertainty"] = True

        # --- expect_refusal?
        if spec.get("expect_refusal", False):
            ok = is_refusal(rec)
            verdict["checks"]["refusal_expected"] = ok
            verdict["pass"] = verdict["pass"] and ok
        else:
            # Should NOT refuse
            ok = is_non_refusal(ans)
            verdict["checks"]["no_unexpected_refusal"] = ok
            verdict["pass"] = verdict["pass"] and ok

        rows.append(verdict)
        total += 1
        if verdict["pass"]:
            passed += 1

    # Print summary
    print("\n=== Score Summary ===")
    for v in rows:
        status = "✅ PASS" if v["pass"] else "❌ FAIL"
        print(f"[{status}] {v['id']}: {v['question']}")
        for k, ok in v["checks"].items():
            print(f"   - {k}: {'OK' if ok else 'FAIL'}")
        if v.get("error"):
            print(f"   - error: {v['error']}")
    print(f"\n{passed}/{total} tests passed")

    # Also write a machine-readable report
    Path("scored_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote scored_results.json")


if __name__ == "__main__":
    main()


