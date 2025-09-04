# Civic LLM — Political Events Chatbot

A reasoning-first chatbot for political & civic questions.  
It interprets user intent, plans evidence queries, retrieves authoritative sources, and writes grounded answers with inline citations.

---

## ✨ Features
- **Reasoning pipeline**: Interpret → Plan → Retrieve → Answer  
- **Model mixing**: different OpenAI models can be used for interpretation, planning, and answering  
- **Summarized history**: keeps conversation context without runaway token usage  
- **Retrieval grounding**: uses Tavily API to fetch evidence from authoritative sites (`.gov`, congress.gov, CRS, etc.)  
- **Adaptive answer styles**: narrative, bullets, comparisons, or party positions depending on the question  
- **Citation enforcement**: factual sentences use inline numeric citations `[1]`  
- **Neutrality**: perspectives are balanced, avoiding bias and templates  
- **Non-political guardrail**: if the question is outside politics, the model politely redirects  

---

## ✅ Requirements
- Python 3.9+  
- OpenAI API key  
- Tavily API key  

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/TanmayShikhare/civic-llm-agent.git
cd civic-llm-agent
bash'''
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add your API keys:
Create a .env file with:

bash
Copy code
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tv-...
🚀 Usage
Run the chatbot locally with Gradio:

bash
Copy code
python3 gradio_app.py
🧪 Evaluation
We provide a lightweight test suite to check reasoning, citations, and refusals.

Run evaluation:

bash
Copy code
python3 run_eval.py --tests tests.yaml --out eval_results.json
Score results:

bash
Copy code
python3 score_eval.py --tests tests.yaml --results eval_results.json
📂 File Structure
bash
Copy code
civic-llm-agent/
├── agent.py          # Core reasoning pipeline (interpret, plan, retrieve, answer)
├── gradio_app.py     # Web UI with Gradio
├── prompts.py        # Prompt templates for interpret & plan
├── tools.py          # Helper functions (Tavily search, reliability scoring, JSON parse)
├── run_eval.py       # Runs evaluation scenarios
├── score_eval.py     # Scores evaluation outputs
├── tests.yaml        # Test scenarios
├── requirements.txt  # Python dependencies
└── README.md         # Documentation
⚠️ Limitations
Focused only on civic & political questions

Retrieval depends on Tavily coverage

Not guaranteed to be fully comprehensive or free of bias

📜 License
MIT License — free to use and modify.

pgsql
Copy code


📜 License

MIT — free to use and modify.


