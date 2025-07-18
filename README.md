# Scholarly Agent Debate App

A Streamlit interface that lets multiple LLM "scholars" debate a topic in
rounds (Position → Critique → Defense) until they reach consensus or you stop
it.  It uses **Pocket Flow** under‑the‑hood to schedule async calls to multiple
provider models (OpenAI, Anthropic, Mistral, Local/Ollama), evaluate
similarity, and run a Judge agent.

---
## Quick Start
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1


# 1. Clone repo & cd in
pipx run python-poetry@^1.8 install  # or ➜ pip install -r requirements.txt

# 2. Set API keys (examples)

# ▸ Bash / Git Bash / WSL
export OPENAI_API_KEY="sk‑..."
export ANTHROPIC_API_KEY="sk‑anth‑..."
export MISTRAL_API_KEY="sk‑ms‑..."

# ▸ Windows PowerShell
setx OPENAI_API_KEY "sk‑..."
setx ANTHROPIC_API_KEY "sk‑anth‑..."

# ▸ Local provider (Ollama)
# Ensure `ollama serve` is running locally.

# 3. Launch
streamlit run streamlit_app.py
```
---

```
## Project layout (virtual):

├── streamlit_app.py            ← UI & run‑loop controller
├── orchestrator.py             ← Debate orchestrator (Pocket Flow graph)
├── agents.py                   ← Agent + Judge definitions
├── providers/                  ← Provider registry & factory
|   ├── __init__.py 
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── mistral_provider.py
│   └── local_provider.py
├── storage.py                  ← JSON session save / load helpers
├── requirements.txt            ← Python deps (incl. Pocket‑Flow)
└── README.md                   ← Install & usage docs
```