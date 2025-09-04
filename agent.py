# agent.py
import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from urllib.parse import urlsplit

from dotenv import load_dotenv

# Load env FIRST
load_dotenv()

from openai import OpenAI

from prompts import INTERPRET_PROMPT, PLAN_PROMPT
from tools import safe_json_parse, tavily_search, reliability_score

# =========================
# Config / Constants
# =========================
_WORD = re.compile(r"[A-Za-z]{4,}")

# Model mixing (env overrides)
MODEL_INTERPRET = os.getenv("OPENAI_MODEL_INTERPRET", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
MODEL_PLAN      = os.getenv("OPENAI_MODEL_PLAN",      os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
MODEL_ANSWER    = os.getenv("OPENAI_MODEL_ANSWER",    os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Fallback order if a call fails (left -> right)
FALLBACK_MODELS = [
    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "gpt-4o-mini",
    "gpt-4o",
]

# History handling
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))     # raw turns kept verbatim (pairs)
SUMMARY_TRIGGER_TURNS = int(os.getenv("SUMMARY_TRIGGER_TURNS", "16"))
HISTORY_SUMMARY_BUDGET_CHARS = int(os.getenv("HISTORY_SUMMARY_BUDGET_CHARS", "1600"))

# Answer behavior knobs
DEFAULT_ANSWER_MAXTOK = int(os.getenv("ANSWER_MAX_TOKENS", "650"))
STRICT_INLINE_CITATIONS = True  # keep per-sentence [#] requirement

# =========================
# OpenAI client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _chat_once(model: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 650) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

def _chat(system: str, user: str, model: str, temperature: float = 0.0, max_tokens: int = 650) -> str:
    """Call OpenAI with graceful fallback across permitted models."""
    tried = []
    last_err = "unknown error"
    for m in [model] + [fm for fm in FALLBACK_MODELS if fm != model]:
        try:
            tried.append(m)
            return _chat_once(m, system, user, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            last_err = str(e)
            # try next fallback
            continue
    # If all fail, return a JSON-like error string for safe parsing by callers
    return f'{{"error":"OpenAI call failed after trying {tried}. Last error: {last_err}"}}'

# =========================
# Scope normalization
# =========================
def _normalize_scope(scope_val: str) -> str:
    s = (scope_val or "").strip().lower()
    if s in ("political", "non-political", "ambiguous"):
        return s
    return "ambiguous"

# =========================
# History normalization & summarization
# =========================
def _normalize_history_list(history: Optional[List]) -> List[Tuple[str, str]]:
    """Coerce arbitrary history into list[(user, assistant)]."""
    if not history:
        return []
    norm: List[Tuple[str, str]] = []
    for t in history:
        if isinstance(t, (list, tuple)):
            if len(t) >= 2:
                u, a = t[0], t[1]
            elif len(t) == 1:
                u, a = t[0], ""
            else:
                u, a = "", ""
        else:
            u, a = str(t), ""
        norm.append((str(u or ""), str(a or "")))
    return norm

def _summarize_turns(turns: List[Tuple[str,str]], budget_chars: int) -> str:
    """
    Summarize many old turns into a compact paragraph.
    We keep it neutral, factual, and short.
    """
    if not turns:
        return ""
    # Build a plain-text transcript to feed the summarizer
    buf = []
    for u, a in turns:
        if u:
            buf.append(f"USER: {u}")
        if a:
            buf.append(f"ASSISTANT: {a}")
    transcript = "\n".join(buf)
    if len(transcript) > budget_chars * 3:
        transcript = transcript[-budget_chars*3:]  # last chunk if extremely long

    sys = (
        "You compress chat context for a political Q&A assistant. "
        "Write a terse, neutral summary of earlier turns (3–6 sentences max). "
        "Do not include advice, opinions, or promises. No citations."
    )
    user = f"Summarize this earlier conversation for context (<= {budget_chars} characters):\n\n{transcript}"
    out = _chat(system=sys, user=user, model=MODEL_INTERPRET, temperature=0.0, max_tokens=300)
    return (out or "").strip()[:budget_chars]

def _format_history_block(history: Optional[List[Tuple[str, str]]]) -> str:
    """
    Returns a compact, model-friendly string:
    - If history length <= MAX_HISTORY_TURNS: include those turns verbatim.
    - If longer: summarize older turns, then include last MAX_HISTORY_TURNS verbatim.
    """
    norm = _normalize_history_list(history)
    if not norm:
        return "Recent conversation: (none)"

    if len(norm) <= MAX_HISTORY_TURNS:
        turns = norm
        summary = ""
    else:
        # Summarize everything except last N turns
        head = norm[:-MAX_HISTORY_TURNS]
        tail = norm[-MAX_HISTORY_TURNS:]
        summary = _summarize_turns(head, HISTORY_SUMMARY_BUDGET_CHARS)
        turns = tail

    lines: List[str] = []
    if summary:
        lines.append("Earlier summary:")
        lines.append(summary)
        lines.append("")

    lines.append("Recent conversation (most recent last):")
    for u, a in turns:
        if u:
            lines.append(f"- USER: {u}")
        if a:
            lines.append(f"- ASSISTANT: {a}")
    return "\n".join(lines)

# =========================
# Pipeline: Interpret
# =========================
def interpret(query: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    t0 = time.time()
    history_block = _format_history_block(history)
    prompt = INTERPRET_PROMPT.format(query=query, history_block=history_block)

    out = _chat(
        system="You are precise and must return strict JSON only.",
        user=prompt,
        model=MODEL_INTERPRET,
        temperature=0.0,
        max_tokens=700,
    )

    data = safe_json_parse(
        out,
        {
            "scope": "ambiguous",
            "intent": "",
            "timeframe": "unknown",
            "jurisdiction": "unknown",
            "entities": [],
            "claims": [],
            "clarification_needed": False,
            "clarification_question": "",
            "redirect_hint": "",
        },
    )
    data["scope"] = _normalize_scope(data.get("scope", ""))
    print(f"[TIMER] interpret: {time.time()-t0:.2f}s")
    return data

# =========================
# Pipeline: Plan
# =========================
def plan(interpret_out: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    # Normalize claims safely
    claims_struct: List[Dict[str, str]] = []
    for c in (interpret_out.get("claims") or []):
        if isinstance(c, dict):
            claims_struct.append(
                {
                    "id": c.get("id", ""),
                    "text": c.get("text", ""),
                    "type": c.get("type", ""),
                }
            )
        elif isinstance(c, str):
            claims_struct.append({"id": "", "text": c, "type": ""})
        else:
            claims_struct.append({"id": "", "text": str(c), "type": ""})

    prompt = PLAN_PROMPT.format(
        claims=json.dumps(claims_struct, ensure_ascii=False),
        timeframe=interpret_out.get("timeframe", "unknown"),
        jurisdiction=interpret_out.get("jurisdiction", "unknown"),
        entities=json.dumps(interpret_out.get("entities", []), ensure_ascii=False),
    )

    out = _chat(
        system="Return ONLY valid JSON. No explanations, no code fences, no markdown, nothing before or after.",
        user=prompt,
        model=MODEL_PLAN,
        temperature=0.0,
        max_tokens=550,
    )

    data = safe_json_parse(out, {"plan_notes": [], "queries": [], "desired_source_types": []})
    print(f"[TIMER] plan: {time.time()-t0:.2f}s")
    return data

# =========================
# Helpers: domain + diversity
# =========================
def _domain(u: str) -> str:
    try:
        return urlsplit(u).netloc.lower()
    except Exception:
        return ""

def diversify_by_domain(
    items: List[Dict[str, Any]], max_per_domain: int = 2, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    buckets = defaultdict(list)
    for it in items:
        buckets[_domain(it.get("url", ""))].append(it)

    diversified: List[Dict[str, Any]] = []
    while True:
        progressed = False
        for dom, bucket in list(buckets.items()):
            if not bucket:
                continue
            if sum(1 for x in diversified if _domain(x["url"]) == dom) >= max_per_domain:
                continue
            diversified.append(bucket.pop(0))
            progressed = True
            if limit and len(diversified) >= limit:
                return diversified
        if not progressed:
            break

    leftovers = [it for b in buckets.values() for it in b]
    if limit:
        diversified.extend(leftovers[: max(0, limit - len(diversified))])
    else:
        diversified.extend(leftovers)
    return diversified

# =========================
# Probes: phrase-first
# =========================
def _extract_phrases(interpret_out: Dict[str, Any], max_phrases: int = 4) -> List[str]:
    phrases: List[str] = []
    for e in (interpret_out.get("entities") or []):
        e = (e or "").strip()
        if e and len(phrases) < max_phrases:
            phrases.append(e)

    for c in (interpret_out.get("claims") or []):
        txt = (c.get("text") if isinstance(c, dict) else str(c)) or ""
        words = re.findall(r"\b[\w'-]+\b", txt)
        if words:
            snippet = " ".join(words[:6]).strip()
            if snippet and len(phrases) < max_phrases:
                phrases.append(snippet)

    seen, out = set(), []
    for p in phrases:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out[:max_phrases]

def _canonical_probes(interpret_out: Dict[str, Any]) -> List[str]:
    year = (interpret_out.get("timeframe") or "").strip()
    year = year if (year.isdigit() and len(year) == 4) else ""

    bases = [
        "site:congress.gov",
        "site:whitehouse.gov",
        "site:crsreports.congress.gov",
        "site:cbo.gov",
        "site:federalregister.gov",
        "site:govinfo.gov",
        "site:gao.gov",
        "site:supremecourt.gov",
        "site:law.cornell.edu",
        "site:justice.gov",
        "site:dhs.gov",
        "site:uscis.gov",
        "site:cbp.gov",
        "site:ice.gov",
        "site:state.gov",
        "site:hhs.gov",
        "site:cms.gov",
        "site:cdc.gov",
        "site:treasury.gov",
        "site:bls.gov",
        "site:census.gov",
    ]

    phrases = _extract_phrases(interpret_out, max_phrases=4)
    if not phrases:
        phrases = [""]

    probes: List[str] = []
    for base in bases:
        for ph in phrases:
            part = f"\"{ph}\"" if ph else ""
            q = " ".join(x for x in [base, year, part] if x).strip()
            if q:
                probes.append(q)

    # a couple of broader phrase-only queries for diversity
    for ph in phrases[:2]:
        q = " ".join(x for x in [year, f"\"{ph}\""] if x).strip()
        if q and q not in probes:
            probes.append(q)

    # Dedup + cap
    seen, final = set(), []
    for p in probes:
        if p not in seen:
            seen.add(p)
            final.append(p)
        if len(final) >= 8:
            break
    return final

# =========================
# Pipeline: Retrieve (Tavily-first, normalized)
# =========================
def _normalize_tavily_items(raw: Any) -> List[Dict[str, str]]:
    """
    Normalize Tavily responses into a list of dicts with keys:
    url, title, snippet, date
    Handles shapes like list[...] or {"results":[...]} or {"data":[...]}.
    """
    if isinstance(raw, dict):
        items = raw.get("results") or raw.get("data") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    norm: List[Dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        norm.append({
            "url": it.get("url") or it.get("link") or "",
            "title": it.get("title") or "",
            "snippet": it.get("snippet") or it.get("content") or "",
            "date": it.get("published_date") or it.get("date") or "",
        })
    return norm

def _search_with_retry(query: str, max_results: int, depth: str, debug: bool) -> List[Dict[str, str]]:
    delays = [0.0, 0.25, 0.5]
    for i, d in enumerate(delays):
        if d:
            time.sleep(d)
        try:
            raw = tavily_search(query, max_results=max_results, search_depth=depth)
            norm = _normalize_tavily_items(raw)
            if debug:
                print(f"[DEBUG] Tavily results for '{query}': {len(norm)}")
            return norm
        except Exception as e:
            print(f"[DEBUG] tavily error try#{i+1} for query: {query} err: {e}")
    return []

def retrieve(interpret_out: Dict[str, Any], plan_out: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    queries: List[str] = list(plan_out.get("queries", []))[:4]
    for p in _canonical_probes(interpret_out):
        if p not in queries:
            queries.append(p)
    queries = queries[:8]

    raw: List[Dict[str, str]] = []
    first = True
    for q in queries:
        raw.extend(_search_with_retry(q, max_results=5, depth="basic", debug=first))
        first = False

    print(f"[DEBUG] queries={len(queries)} raw_hits={len(raw)}")
    if raw:
        print("[DEBUG] raw sample:", raw[0])

    # Dedup + reliability
    seen = set()
    results: List[Dict[str, Any]] = []
    for r in raw:
        u = (r.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        try:
            score = float(reliability_score(u) or 0.0)
        except Exception:
            score = 0.0
        results.append(
            {
                "title": r.get("title", ""),
                "url": u,
                "snippet": r.get("snippet", ""),
                "reliability_score": score,
                "reliability": "High" if score >= 0.90 else ("Medium" if score >= 0.75 else "Low"),
                "date": r.get("date", ""),
            }
        )

    # cautious timeframe filter
    timeframe = (interpret_out.get("timeframe") or "").strip()
    if timeframe.isdigit():
        year = timeframe
        pre_count = len(results)
        filtered = [
            r for r in results
            if (year in (r.get("title") or ""))
            or (year in (r.get("snippet") or ""))
            or (year in (r.get("date") or ""))
        ]
        if filtered and len(filtered) >= max(2, pre_count // 3):
            results = filtered

    # diversify
    results.sort(key=lambda x: x.get("reliability_score", 0.0), reverse=True)
    results = diversify_by_domain(results, max_per_domain=2, limit=None)

    print(f"[DEBUG] dedup_results={len(results)} (after diversity interleave)")

    evidence_table: List[Dict[str, Any]] = []
    for r in results:
        evidence_table.append(
            {
                "claim_id": "",
                "url": r["url"],
                "title": r["title"],
                "snippet": r["snippet"],
                "reliability": r["reliability"],
            }
        )

    print(f"[TIMER] retrieve: {time.time()-t0:.2f}s")
    return {"evidence_table": evidence_table}

# =========================
# Style heuristics for answers
# =========================
def _style_from_intent(intent: str) -> str:
    """Return one of: 'bullets', 'narrative', 'compare', 'party'."""
    s = (intent or "").lower()
    # explicit party stance ask (both)
    if any(k in s for k in ["both parties", "democrats and republicans", "party positions", "gop vs", "compare parties", "democratic vs republican"]):
        return "party"
    # general comparisons
    if any(k in s for k in ["compare", "difference", "versus", "vs.", "pros and cons"]):
        return "compare"
    # list/bullets cues
    if any(k in s for k in ["key points", "bullets", "list", "top points", "highlights", "summary points"]):
        return "bullets"
    # explanation cues
    if any(k in s for k in ["why", "how", "explain", "overview", "what happened"]):
        return "narrative"
    return "narrative"

def _length_from_intent(intent: str) -> str:
    """Return 'short', 'medium', or 'long' based on cues."""
    s = (intent or "").lower()
    if any(k in s for k in ["brief", "short", "tl;dr", "quick"]):
        return "short"
    if any(k in s for k in ["deep", "detailed", "long", "comprehensive", "in depth", "in-depth"]):
        return "long"
    return "medium"

def _tokens_for_length(length: str) -> int:
    base = globals().get("DEFAULT_ANSWER_MAXTOK", 650)
    if length == "short":
        return min(350, base)
    if length == "long":
        return min(900, base)
    # medium
    return base

# NEW: detect single-party focus
def _party_focus(intent: str) -> Optional[str]:
    """
    Detect if the user wants ONLY one party's view.
    Returns 'democratic', 'republican', or None.
    """
    text = (intent or "").lower()

    # If they explicitly ask for BOTH, don't single-focus
    if any(k in text for k in ["both parties", "democrats and republicans", "party positions", "dem vs gop", "democratic vs republican"]):
        return None

    # Single-party checks
    if any(k in text for k in ["democratic party", "democrat position", "democratic position", "democrats' position", "dems position", "what did democrats"]):
        return "democratic"

    if any(k in text for k in ["republican party", "gop position", "republican position", "republicans' position", "what did republicans"]):
        return "republican"

    return None

# NEW: map length hint to sentence range
def _sentence_range(length_hint: str) -> Tuple[int, int]:
    if length_hint == "short":
        return (2, 4)
    if length_hint == "long":
        return (8, 12)
    return (4, 7)  # medium

def _topic_is_dynamic(intent: str, timeframe: str) -> bool:
    s = (intent or "").lower()
    tf = (timeframe or "").lower()
    hot_words = [
        "election", "primary", "campaign", "current debate", "ongoing",
        "breaking", "live", "latest", "today", "this week", "this month"
    ]
    if any(w in s for w in hot_words):
        return True
    # if timeframe is current/near-current year, also treat as dynamic
    return any(y in tf for y in ["2023", "2024", "2025"])


# =========================
# Pipeline: Answer (adaptive style)
# =========================
def answer(interpret_out: Dict[str, Any], evidence_table: List[Dict[str, Any]]) -> str:
    t0 = time.time()

    # Non-political: let the model say it naturally (no rigid template)
    scope = (interpret_out.get("scope") or "").strip().lower()
    if scope == "non-political":
        sys = (
            "You are a helpful assistant. If the user's question is not political/civic, "
            "gently explain that this chatbot focuses on politics/government, "
            "then briefly suggest how they could reframe it to fit (one sentence). "
            "Be polite and concise; don't continue a long off-topic conversation."
        )
        user = f"The user asked: {interpret_out.get('intent','(no intent)')}\nWrite a brief, natural response (no lists)."
        out = _chat(system=sys, user=user, model=MODEL_ANSWER, temperature=0.2, max_tokens=160)
        print(f"[TIMER] answer: {time.time()-t0:.2f}s")
        return out

    # Clarification needed?
    if interpret_out.get("clarification_needed") and (interpret_out.get("clarification_question") or "").strip():
        print(f"[TIMER] answer: {time.time()-t0:.2f}s")
        return (interpret_out.get("clarification_question") or "").strip()

    # Unique URLs from evidence
    uniq_urls: List[str] = []
    seen = set()
    for row in (evidence_table or []):
        if not isinstance(row, dict):
            continue
        u = (row.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            uniq_urls.append(u)

    if not uniq_urls:
        print(f"[TIMER] answer: {time.time()-t0:.2f}s")
        return (
            "I couldn’t find sufficiently grounded sources to answer confidently. "
            "Try narrowing the timeframe or jurisdiction, or naming the key entities."
        )

    uniq_urls = uniq_urls[:12]

    # Build evidence listing for the prompt
    ev_lines: List[str] = []
    highs, meds = 0, 0
    for r in (evidence_table or [])[:24]:
        rel = (r.get("reliability") or "")
        if rel == "High":
            highs += 1
        elif rel == "Medium":
            meds += 1
        ev_lines.append(f"- {r.get('title','')} | {r.get('url','')} | {rel}")

    source_list = "\n".join(f"[{i+1}] {u}" for i, u in enumerate(uniq_urls))

    # Confidence cue (internal)
    official_hits = sum(
        1 for u in uniq_urls if any(host in u for host in (
            "congress.gov","whitehouse.gov","crsreports.congress.gov","cbo.gov",
            "supremecourt.gov","law.cornell.edu",".gov",".mil",
        ))
    )
    if official_hits >= 2:
        confidence_hint = "Evidence includes multiple official/primary sources."
    elif (highs + meds) >= 2:
        confidence_hint = "Evidence includes a mix of medium/high-reliability sources."
    else:
        confidence_hint = "Evidence is limited or lower-confidence; express uncertainty."

        # --- Confidence & uncertainty cues ---
    evidence_is_thin = (len(uniq_urls) < 2) or ((highs + meds) < 2)

    user_intent = (interpret_out.get("intent") or "")
    timeframe = interpret_out.get("timeframe", "unknown")
    topic_dynamic = _topic_is_dynamic(user_intent, timeframe)

    must_note_limits = topic_dynamic or evidence_is_thin
    limitations_line = (
        "Limitations: This topic evolves quickly and details can change; treat this as a snapshot."
        if must_note_limits else ""
    )


    # Style & length from intent
    user_intent = (interpret_out.get("intent") or "")
    style = _style_from_intent(user_intent)
    length_hint = _length_from_intent(user_intent)
    max_toks = _tokens_for_length(length_hint)
    lo, hi = _sentence_range(length_hint)
    focus_party = _party_focus(user_intent)  # 'democratic' | 'republican' | None

    # System prompt: adaptive style
    citation_rule = (
        "- EVERY factual sentence MUST include an inline numeric citation [#] tied to the Source list below.\n"
        if STRICT_INLINE_CITATIONS else
        "- Include citations where appropriate.\n"
    )

    sys = (
        "You are a neutral political analyst.\n"
        "- Be accurate, concise, and even-handed.\n"
        f"{citation_rule}"
        "- Avoid persuasive adjectives; focus on what happened, when, who, and what sources say.\n"
        "- Keep timelines and outcomes precise when supported by sources.\n"
        "- If evidence is limited or mixed, acknowledge uncertainty briefly.\n"
    )

    # Style instruction (UPDATED)
    if focus_party:
        party_label = "Democratic position" if focus_party == "democratic" else "Republican position"
        style_block = (
            "If context is absolutely necessary, begin with ONE short neutral sentence to set the scene; "
            "otherwise go straight to the party section.\n"
            f"Then write **{party_label}** in about {lo}–{hi} complete sentences. "
            "Stick to sourced facts, policy statements, and actions—avoid speculation. "
            "Every factual sentence must include [#] citations."
        )
    else:
        if style == "party":
            style_block = (
                f"Start with a brief neutral summary ({max(lo-1,1)}–{max(2,lo)} sentences). "
                f"Then add two labeled subsections:\n"
                f"  - Democratic position ({lo}–{hi} sentences)\n"
                f"  - Republican position ({lo}–{hi} sentences)\n"
                "Keep the two subsections parallel and factual. Each sentence has a [#] citation."
            )
        elif style == "compare":
            style_block = (
                f"Write a brief neutral summary ({max(lo-1,1)}–{max(2,lo)} sentences), "
                "then a compare list of 3–6 bullets highlighting concrete differences and overlaps; "
                "each bullet is one complete sentence with a [#] citation."
            )
        elif style == "bullets":
            style_block = (
                f"Return a tight bullet list of {min(8, max(5, lo))}–{min(10, hi)} items covering the key facts first; "
                "each bullet is a complete sentence with a [#] citation."
            )
        else:  # narrative
            style_block = (
                "Return a compact narrative overview (1–3 short paragraphs). "
                "Keep it focused and readable; every factual sentence has a [#] citation."
            )

    prompt = (
        f"User intent: {user_intent}\n"
        f"Timeframe: {interpret_out.get('timeframe','unknown')}\n"
        f"Confidence cue (internal): {confidence_hint}\n\n"
        "Evidence (title | url | reliability):\n" + "\n".join(ev_lines) +
        "\n\nWrite the answer with the following style:\n" + style_block + "\n\n" +
        (f"{limitations_line}\n\n" if limitations_line else "") +
        "Source list:\n" + source_list
    )

    txt = _chat(system=sys, user=prompt, model=MODEL_ANSWER, temperature=0.15, max_tokens=max_toks)
    print(f"[TIMER] answer: {time.time()-t0:.2f}s")
        # Post-check: enforce inline citations if required
    if STRICT_INLINE_CITATIONS and not re.search(r"\[\d+\]", txt or ""):
        fix_sys = (
            "Add inline numeric citations [#] to all factual sentences, "
            "mapping to the Source list as provided. Do not change meaning."
        )
        fix_user = (
            "Rewrite the same answer text, but ensure each factual sentence "
            "has [#] citations aligned to the Source list below.\n\n"
            f"Answer:\n{txt}\n\nSource list stays the same:\n{source_list}"
        )
        txt2 = _chat(
            system=fix_sys,
            user=fix_user,
            model=MODEL_ANSWER,
            temperature=0.0,
            max_tokens=max_toks,
        )
        if txt2 and re.search(r"\[\d+\]", txt2):
            txt = txt2

    return txt

# =========================
# Wrapper for app / gradio
# =========================
def interpret_then_plan(query: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """Run Interpret → Plan, with conversation context (summarized if long)."""
    interp = interpret(query, history=history)
    result: Dict[str, Any] = {
        "interpret": interp,
        "plan": {"plan_notes": [], "queries": [], "desired_source_types": []},
    }
    if interp.get("scope") == "non-political":
        return result
    result["plan"] = plan(interp)
    return result





