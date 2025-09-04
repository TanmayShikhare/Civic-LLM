# tools.py
import os
import json
import re
from typing import Any, Dict, List
import requests
from urllib.parse import urlsplit

JSON_LIKE_BRACES = re.compile(r'^[^{\[]*({|\[).*', re.DOTALL)

def safe_json_parse(s: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse strict JSON. If it fails, try to trim to the first JSON object/array.
    Always return a dict (fallback if needed).
    """
    if not isinstance(s, str):
        return dict(fallback)
    s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return dict(fallback)
    except Exception:
        pass
    # Try to recover JSON object/array substring
    m = JSON_LIKE_BRACES.search(s)
    if m:
        candidate = s[m.start():]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return dict(fallback)

def tavily_search(query: str, max_results: int = 5, search_depth: str = "basic", include_domains: List[str] = None) -> List[Dict[str, str]]:
    """
    Tavily Search API wrapper (free tier supported). Returns a list of dicts with title, url, snippet, date.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in environment.")
    payload: Dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,   # "basic" or "advanced"
        "include_answer": False,
        "include_images": False,
        "include_domains": include_domains or [],
    }
    resp = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    out: List[Dict[str, str]] = []
    for r in data.get("results", []):
        out.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": r.get("content") or "",
            "date": r.get("published_date") or "",
        })
    return out

def reliability_score(url: str) -> float:
    """
    Lightweight, content-agnostic heuristic favoring official/primary sources.
    No topic keywords. Purely domain-based.
    """
    try:
        host = urlsplit(url).netloc.lower()
    except Exception:
        host = ""
    if not host:
        return 0.1

    # Top tier: official / primary
    if host.endswith(".gov") or host.endswith(".mil"):
        return 0.98
    if host.endswith(".edu"):
        return 0.92
    primary_whitelist = {
        "congress.gov", "whitehouse.gov", "crsreports.congress.gov", "cbo.gov",
        "federalregister.gov", "govinfo.gov", "gao.gov", "supremecourt.gov",
        "law.cornell.edu", "justice.gov",
        "dhs.gov", "uscis.gov", "cbp.gov", "ice.gov", "state.gov",
        "hhs.gov", "cms.gov", "cdc.gov",
        "treasury.gov", "bls.gov", "census.gov",
    }
    if host in primary_whitelist:
        return 0.96

    # High-quality news/nonpartisan outlets
    hq = {
        "apnews.com", "reuters.com", "pbs.org", "npr.org", "pewresearch.org",
        "cfr.org", "brookings.edu",
        "nytimes.com", "wsj.com", "washingtonpost.com", "economist.com",
        "bbc.com", "ft.com", "politico.com", "axios.com"
    }
    if any(host.endswith(h) for h in hq):
        return 0.88

    # Default
    return 0.65








