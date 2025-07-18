import asyncio, json, datetime, os
import streamlit as st
from orchestrator import DebateConfig, DebateOrchestrator
from storage import save_session, load_session

st.set_page_config(page_title="Scholarly Agent Debate", layout="wide")

# ------------- Sidebar settings ----------------
st.sidebar.header("Agent Config")

def default_agents():
    return [
        {"name": "Alice‑OpenAI", "provider_name": "openai", "model": "gpt-4o-mini"},
        {"name": "Bob‑Anthropic", "provider_name": "anthropic", "model": "claude-3-sonnet-20240229"},
        {"name": "Cora‑Mistral", "provider_name": "mistral", "model": "mistral-large-latest"},
        {"name": "Dave‑Local", "provider_name": "local", "model": "llama3"},
    ]

a_num = st.sidebar.number_input("# Agents", min_value=2, max_value=8, value=4)
if "agent_cfgs" not in st.session_state:
    st.session_state.agent_cfgs = default_agents()[:a_num]

for i in range(a_num):
    cfg = st.session_state.agent_cfgs[i]
    st.sidebar.text_input(f"Agent {i+1} Name", value=cfg["name"], key=f"name{i}")
    st.sidebar.selectbox(
        "Provider", ["openai", "anthropic", "mistral", "local"],
        index=["openai","anthropic","mistral","local"].index(cfg["provider_name"]),
        key=f"prov{i}")
    st.sidebar.text_input("Model", value=cfg["model"], key=f"model{i}")

judge_model = st.sidebar.text_input("Judge Model (OpenAI)", value="gpt-4o-mini")

st.sidebar.markdown("---")
auto_run = st.sidebar.checkbox("Auto‑advance rounds", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("Save Session") and "orch" in st.session_state:
    path = f"debate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_session(path, st.session_state.orch)
    st.sidebar.success(f"Saved → {path}")

if st.sidebar.button("Load Session"):
    uploaded = st.sidebar.file_uploader("Choose .json", type="json")
    if uploaded:
        data = json.load(uploaded)
        st.write(data)

# ------------- Main UI ----------------
st.title("📚 Scholarly Agent Debate")

user_topic = st.text_input("Enter a topic or question:")

if st.button("Start Debate") and user_topic:
    # Build config from sidebar widgets
    cfgs = []
    for i in range(a_num):
        cfgs.append({
            "name": st.session_state[f"name{i}"],
            "provider_name": st.session_state[f"prov{i}"],
            "model": st.session_state[f"model{i}"],
        })
    judge_cfg = {"name": "Judge", "provider_name": "openai", "model": judge_model}
    conf = DebateConfig(cfgs, judge_cfg, auto_run)
    st.session_state.orch = DebateOrchestrator(conf)
    st.session_state.topic = user_topic
    st.session_state.run_task = asyncio.create_task(st.session_state.orch.next_round(user_topic))

# Async poll helper
async def poll_loop():
    while True:
        from typing import Optional
        orch: Optional[DebateOrchestrator] = st.session_state.get("orch")
        if not orch: break
        if orch.stopped: break
        await asyncio.sleep(0.1)

if "orch" in st.session_state:
    orch = st.session_state.orch
    # Display transcripts
    for ag in orch.agents:
        with st.expander(f"{ag.name}"):
            for turn in ag.transcript:
                st.markdown(f"**{turn['round'].capitalize()}** → {turn['content']}")
    # Verdicts
    for item in orch.history:
        st.info(f"Round {item['round']} Judge: {item['verdict']}")

    # Controls
    col1, col2 = st.columns(2)
    if col1.button("Advance Round") and not orch.config.auto and not orch.stopped:
        asyncio.create_task(orch.next_round(st.session_state.topic))
    if col2.button("Stop Debate"):
        orch.stopped = True

    st.write("\n---\n")
    st.write(f"**Phase:** {orch.phase} • **Round:** {orch.round_num}")

    # Inject evidence box
    new_evidence = st.text_area("Inject new evidence (shared with all agents):", height=100)
    if st.button("Submit Evidence") and new_evidence:
        # Prepend evidence to topic for next round
        st.session_state.topic = new_evidence + "\n\n" + st.session_state.topic
        st.success("Evidence added. It will be included in next round prompts.")