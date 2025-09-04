# prompts.py

INTERPRET_PROMPT = """You are assessing a user query for political analysis.

{history_block}

Reason about scope using concepts, not keywords. Consider whether answering requires government policy, law, elections, legislative/judicial context, or actions of public officials/agencies. Do not use keyword rules.

Return the following fields:
1) scope — one of "political", "non-political", "ambiguous" (EXACT spelling)
2) intent — one concise line paraphrasing what the user wants
3) timeframe — a specific year or range if clearly implied; otherwise "unknown"
4) jurisdiction — e.g., "U.S. federal", state/country name, or "unknown"
5) entities — salient proper nouns only (people, offices, laws, courts, agencies)
6) claims — 3–6 short, testable, SINGLE-proposition items. Each claim must have:
   - id: "c1", "c2", … (stable labels)
   - text: precise wording suitable for verification (avoid hedging)
   - type: one of "fact", "procedure", "position", or "outcome"
7) clarification_needed — true only if you cannot proceed without one key fact
8) clarification_question — the single, essential question if clarification_needed=true; else ""
9) redirect_hint — short, helpful hint if scope="non-political"; else ""

Guidelines:
- Infer timeframe/jurisdiction cautiously. If not clear, use "unknown".
- Avoid stale anchors like "as of <year>" unless the user explicitly anchors time.
- Do not include explanations or extra fields.

Return ONLY a single JSON object with EXACT keys:
{{
  "scope": "...",
  "intent": "...",
  "timeframe": "...",
  "jurisdiction": "...",
  "entities": ["..."],
  "claims": [{{"id":"c1","text":"...","type":"fact"}}],
  "clarification_needed": false,
  "clarification_question": "",
  "redirect_hint": ""
}}

STRICT FORMAT INSTRUCTION: Return ONLY the JSON object. No prose, no code fences, nothing before or after.

User query:
{query}
"""

PLAN_PROMPT = """You are creating an evidence plan to verify claims and ground an answer.

Claims: {claims}
Timeframe: {timeframe}
Jurisdiction: {jurisdiction}
Entities: {entities}

Goals:
- Prefer primary/official sources when feasible (e.g., congress.gov, whitehouse.gov, crsreports.congress.gov, cbo.gov, relevant agency .gov). Include high-quality nonpartisan/academic as needed.
- Formulate search queries using SHORT QUOTED PHRASES (multi-word) derived from claims/entities, not single-word keyword lists.
- Incorporate timeframe/jurisdiction naturally when it helps precision (e.g., a year).
- Keep queries specific enough to retrieve authoritative pages (bill text, official releases, court docs, agency fact sheets, CRS/CBO).
- Avoid topic-specific hardcoded keywords; rely on the content of the claims.

Return ONLY a single JSON object with exactly these keys and array values:
{{
  "plan_notes": [
    "c1 → what evidence best resolves it (e.g., official bill summary, signed law, agency release, court opinion)",
    "c2 → ...",
    "c3 → ..."
  ],
  "queries": [
    "site:congress.gov \\"<short phrase from claim/entity>\\" <year_if_helpful>",
    "site:whitehouse.gov \\"<short phrase>\\" <year_if_helpful>",
    "\\"<short phrase>\\" <year_if_helpful>",
    "site:crsreports.congress.gov \\"<short phrase>\\""
  ],
  "desired_source_types": [
    "official.gov",
    "non-partisan think tanks",
    "major outlets",
    "academic",
    "court opinions"
  ]
}}

Notes:
- Provide 3–6 queries total. Use quotes around phrases. Keep each query concise.
- Do not add extra fields or prose.

STRICT FORMAT INSTRUCTION: Return ONLY the JSON object. No prose, no code fences, nothing before or after.
"""








