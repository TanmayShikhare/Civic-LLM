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

