# Civic LLM — Political Events Chatbot  

A **reasoning-first chatbot** that answers questions about political and civic events.  
It interprets user intent, plans evidence queries, retrieves authoritative sources, and produces **grounded answers with inline citations**.  

---

## ✨ Features  

- **Reasoning-first pipeline**: Interpret → Plan → Retrieve → Answer  
- **Model mixing**: use different OpenAI models for interpretation, planning, and answering  
- **Summarized conversation history**: preserves context without runaway token usage  
- **Retrieval grounding**: uses the Tavily API to fetch evidence from authoritative sites (`.gov`, congress.gov, CRS, etc.)  
- **Adaptive answer styles**: narrative, bullets, comparisons, or party positions depending on the query  
- **Citation enforcement**: every factual claim carries inline numeric citations `[1]`  
- **Neutrality**: perspectives are balanced, avoiding bias or rigid templates  
- **Non-political guardrail**: if the question is outside politics, the model politely declines and suggests reframing  

---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/TanmayShikhare/civic-llm-agent.git
cd civic-llm-agent


Install dependencies:
pip install -r requirements.txt

Create a .env file in the repo root:

OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tv-...

# Optional model overrides
OPENAI_MODEL=gpt-4o-mini
OPENAI_MODEL_INTERPRET=gpt-4o-mini
OPENAI_MODEL_PLAN=gpt-4o-mini
OPENAI_MODEL_ANSWER=gpt-4o-mini

# Optional behavior knobs
MAX_HISTORY_TURNS=12
SUMMARY_TRIGGER_TURNS=16
HISTORY_SUMMARY_BUDGET_CHARS=1600
ANSWER_MAX_TOKENS=650

🚀 Usage (Gradio UI)
python3 gradio_app.py

🧪 Evaluation

Run scenarios and save raw outputs:
python3 run_eval.py --tests tests.yaml --out eval_results.json

Score them:
python3 score_eval.py --tests tests.yaml --results eval_results.json

🔧 Configuration Notes

Models can be mixed via env vars (OPENAI_MODEL_*).

History is summarized beyond MAX_HISTORY_TURNS to control cost.

Citations are enforced post-generation if missing.

📂 File Structure
civic-llm-agent/
├── agent.py          # Core pipeline (interpret, plan, retrieve, answer)
├── gradio_app.py     # Simple Gradio UI (chat + sources)
├── prompts.py        # Prompt templates for interpret & plan
├── tools.py          # Tavily wrapper, reliability scoring, safe JSON parser
├── run_eval.py       # Runs evaluation scenarios
├── score_eval.py     # Scores eval outputs (citations, refusal, uncertainty)
├── tests.yaml        # Test scenarios
├── requirements.txt  # Python dependencies
└── README.md         # This file

🛠️ Troubleshooting

Missing API key: set OPENAI_API_KEY and TAVILY_API_KEY in .env, then restart.

Citations missing: the answer step post-checks and repairs; if it still fails, tighten the query or provide clearer timeframe/entities.

Slow retrieval: Tavily calls are sequential for reliability—reduce queries in agent.py if needed.

⚠️ Limitations

Focused on civic/political topics (will gently refuse off-topic).

Retrieval quality depends on Tavily coverage.

Not a substitute for legal advice or official guidance.

📜 License

MIT — free to use and modify.


