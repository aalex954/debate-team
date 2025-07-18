# Scholarly Agent Debate App

A Streamlit interface that lets multiple LLM "scholars" debate a topic in
rounds (Position → Critique → Defense) until they reach consensus or you stop
it.  It uses **Pocket Flow** under‑the‑hood to schedule async calls to multiple
provider models (OpenAI, Anthropic, Mistral, Local/Ollama), evaluate
similarity, and run a Judge agent.

---
## Quick Start
```bash
# Clone the repository
git clone https://github.com/aalex954/yeet-agent.git
cd yeet-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch App
streamlit run streamlit_app.py
```

### API Key Setup

You need at least one API key from the supported providers:

#### Linux/Mac
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MISTRAL_API_KEY="..."
```

#### Windows PowerShell (Temporary)
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:MISTRAL_API_KEY = "..."
```

#### Windows PowerShell (Persistent)
```powershell
setx OPENAI_API_KEY "sk-..."
setx ANTHROPIC_API_KEY "sk-ant-..."
setx MISTRAL_API_KEY "..."
```
> **Note**: Close and reopen your terminal after using `setx` to pick up the new variables.

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