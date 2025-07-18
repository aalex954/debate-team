# Copilot Instructions for Scholarly Agent Debate App

## Project Overview
- This app orchestrates multi-agent LLM debates in rounds (Position → Critique → Defense) using Streamlit for UI and PocketFlow for async orchestration.
- Agents (LLMs) and a Judge agent interact via provider wrappers (OpenAI, Anthropic, Mistral, Local/Ollama).
- Consensus is reached via similarity scoring and a Judge verdict.

## Key Components
- `streamlit_app.py`: Main UI, run-loop, and session management. Entry point for users.
- `orchestrator.py`: Core debate logic, round management, and PocketFlow async graph construction.
- `agents.py`: Defines `Agent` and `Judge` classes. Agents use providers to generate responses and track transcripts.
- `providers/`: Registry and wrappers for LLM APIs. Add new providers by registering a class in `__init__.py`.
- `storage.py`: Session save/load helpers (JSON format).

## Developer Workflows
- **Setup**: Use Python 3.10+ and create a virtual environment. Install dependencies with `pip install -r requirements.txt`.
- **Run**: Launch with `streamlit run streamlit_app.py`.
- **API Keys**: Set environment variables for each provider (see README for details).
- **Session Save/Load**: Use sidebar buttons in the UI to save/load debate sessions.
- **Debugging**: Most errors surface in the Streamlit UI. For backend debugging, add print/log statements in `orchestrator.py` and `agents.py`.

## Patterns & Conventions
- **Agent Construction**: Agents are created from config dicts and use a provider factory (`providers.create`).
- **Async Orchestration**: Debate rounds are modeled as PocketFlow async graphs. Each agent's turn is a node; chaining is done via `.next()`.
- **Judge Agent**: Receives the full debate state as JSON and returns a structured verdict dict.
- **Similarity**: Uses `sentence-transformers` for agent response similarity.
- **Provider Extensibility**: To add a new LLM provider, subclass `Provider`, implement `complete`, and register with `@register`.
- **Session State**: Streamlit's `st.session_state` is used for all persistent UI state.

## Integration Points
- **PocketFlow**: Used for async orchestration. If unavailable, code falls back to simple async execution.
- **Providers**: All LLM calls are routed through provider classes. API keys must be set in the environment.
- **Judge**: Always uses OpenAI by default, but can be configured.

## Example: Adding a New Provider
```python
from providers import Provider, register

@register("myprovider")
class MyProvider(Provider):
    async def complete(self, prompt: str) -> str:
        # Implement API call here
        return "response"
```

## Example: Customizing Debate Flow
- To change round types or add new phases, update `_build_flow` in `orchestrator.py`.
- Each agent's turn is a node; chain nodes with `.next()`.

## References
- See `README.md` for setup and usage.
- See `orchestrator.py` for debate orchestration and PocketFlow usage.
- See `agents.py` for agent logic and transcript management.
- See `providers/` for provider patterns.

---
_If any section is unclear or missing, please specify what needs improvement or what workflows you want documented further._
