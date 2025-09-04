Civic LLM — Political Events Chatbot

A reasoning-first chatbot that answers questions about political and civic events.
It interprets user intent, plans evidence queries, retrieves authoritative sources, and produces grounded answers with inline citations.

✨ Features

Reasoning-first pipeline: Interpret → Plan → Retrieve → Answer.

Model mixing: different OpenAI models can be used for interpretation, planning, and answering.

Summarized conversation history: keeps context without runaway token usage.

Retrieval grounding: uses Tavily API to fetch evidence from authoritative sites (.gov, congress.gov, crsreports, etc.).

Adaptive answer styles: narrative, bullets, comparisons, or party positions depending on the question.

Citation enforcement: every factual claim carries inline numeric citations [1].

Neutrality: perspectives are balanced, avoiding bias and templates.

Non-political guardrail: if the question is outside politics, the model gently declines and suggests reframing.

⚙️ Installation

Clone the repository:
git clone https://github.com/TanmayShikhare/civic-llm-agent.git
cd civic-llm-agent

Install dependencies:
pip install -r requirements.txt

Add your API keys:
Create a .env file with:

OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tv-...

🚀 Usage

Run the chatbot locally with Gradio:
python3 gradio_app.py

🧪 Evaluation

We provide a lightweight test suite to check reasoning, citations, and refusals.

Run evaluation:
python3 run_eval.py --tests tests.yaml --out eval_results.json

Score results:
python3 score_eval.py --tests tests.yaml --results eval_results.json


📂 File Structure

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

Focused only on civic & political questions.

Retrieval depends on Tavily coverage.

Not guaranteed to be fully comprehensive or free of bias.

📜 License

MIT License
 — free to use and modify.