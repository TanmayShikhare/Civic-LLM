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

Clone the repository, install dependencies, and set API keys:

```bash
git clone https://github.com/TanmayShikhare/civic-llm-agent.git
cd civic-llm-agent
pip install -r requirements.txt

# Create a .env file with:
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tv-...



