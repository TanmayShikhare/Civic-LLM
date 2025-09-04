# app.py
import json
import time
from agent import interpret, plan, retrieve, answer
from context import Context

CTX = Context()

def run_one(user_q: str):
    t0 = time.time()

    # 1) Interpret this turn
    interp = interpret(user_q)

    # Clarification gate
    if interp.get("clarification_needed") and (interp.get("clarification_question") or "").strip():
        print("\n=== CLARIFICATION NEEDED ===")
        print(interp["clarification_question"].strip())
        return

    # Boundary guard
    if interp.get("scope") == "non-political":
        print("\n=== ANSWER ===")
        print(answer(interp, []))
        print(f"\n[TIMER] total: {time.time() - t0:.2f}s")
        return

    # 2) Router: refinement vs new topic; repair if big shift
    if CTX.is_refinement(interp):
        need_fix, repair_msg = CTX.needs_repair(interp)
        if need_fix:
            print("\n=== CONVERSATIONAL REPAIR ===")
            print(repair_msg)
            return
        merged_interp = CTX.merge_with_interpret(interp)
        pl = plan(merged_interp)
        cur_interp = merged_interp
    else:
        CTX.update_from_interpret(interp)
        pl = plan(interp)
        cur_interp = interp

    # 3) Debug views
    print("\n=== CONTEXT ===")
    print(CTX.summary())
    print("\n=== INTERPRET ===")
    print(json.dumps(cur_interp, indent=2, ensure_ascii=False))
    print("\n=== PLAN ===")
    print(json.dumps(pl, indent=2, ensure_ascii=False))

    # 4) Retrieve + Answer
    ev = retrieve(cur_interp, pl)
    rows = ev.get("evidence_table", [])

    # Evidence sample
    print("\n=== EVIDENCE (top 5) ===")
    for row in rows[:5]:
        print({
            "claim_id": row.get("claim_id", ""),
            "reliability": row.get("reliability", ""),
            "title": (row.get("title", "") or "")[:120],
            "url": row.get("url", "")
        })

    print("\n=== ANSWER ===")
    try:
        ans = answer(cur_interp, rows)
        print(ans)

        # Build meta and persist turn
        uniq_urls, seen = [], set()
        for r in rows:
            u = r.get("url")
            if u and u not in seen:
                uniq_urls.append(u); seen.add(u)
        CTX.record_turn(
            user_text=user_q,
            interp=cur_interp,
            plan=pl,
            answer_meta={"citations": uniq_urls[:12]}
        )
        # On success, update context with whatever (merged) interp we used
        CTX.update_from_interpret(cur_interp)
    except Exception as e:
        print(f"Answering failed: {e}")

    print(f"\n[TIMER] total: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    print("Civic LLM — CLI\n"
          "Enter to quit • type '/reset' to clear context • type '/ctx' to view context summary")
    while True:
        q = input("\nAsk a question: ").strip()
        if not q:
            break
        if q.lower() == "/reset":
            CTX.reset()
            print("Context reset.")
            continue
        if q.lower() == "/ctx":
            print(CTX.summary())
            continue
        try:
            run_one(q)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
