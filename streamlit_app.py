import asyncio, json, datetime, os
import streamlit as st
from orchestrator import DebateConfig, DebateOrchestrator
from storage import save_session, load_session

st.set_page_config(page_title="Scholarly Agent Debate", layout="wide")

# ------------- Sidebar settings ----------------
st.sidebar.header("Agent Config")

def default_agents():
    return [
        {"name": "OpenAI", "provider_name": "openai", "model": "gpt-4o-mini"},
        {"name": "Mistral", "provider_name": "mistral", "model": "mistral-large-latest"},
    ]

a_num = st.sidebar.number_input("# Agents", min_value=2, max_value=8, value=2)  # Default to 2 agents
if "agent_cfgs" not in st.session_state:
    st.session_state.agent_cfgs = default_agents()[:a_num]

for i in range(a_num):
    cfg = st.session_state.agent_cfgs[i]
    st.sidebar.text_input(f"Agent {i+1} Name", value=cfg["name"], key=f"name{i}")
    provider_options = ["openai", "anthropic", "mistral", "local"]
    st.sidebar.selectbox(
        "Provider", provider_options,
        index=provider_options.index(cfg["provider_name"]),
        key=f"prov{i}")
    # Model dropdown options per provider
    model_options = {
        "openai": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"],
        "mistral": ["mistral-large-latest", "mistral-medium", "mistral-small"],
        "local": ["llama3", "llama2", "mistral-7b", "custom"]
    }
    selected_provider = st.session_state.get(f"prov{i}", cfg["provider_name"])
    st.sidebar.selectbox(
        "Model", model_options.get(selected_provider, [cfg["model"]]),
        index=model_options.get(selected_provider, [cfg["model"]]).index(cfg["model"]) if cfg["model"] in model_options.get(selected_provider, [cfg["model"]]) else 0,
        key=f"model{i}")

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
    
    # Use asyncio.run() instead of create_task
    import nest_asyncio
    nest_asyncio.apply()
    
    # Run in a way that's compatible with Streamlit
    try:
        # For first round only
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(st.session_state.orch.next_round(user_topic))
    except Exception as e:
        st.error(f"Error starting debate: {e}")

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
    
    # Helper function to create one-line summaries
    def get_summary(text, max_length=100):
        # Get first sentence or first part of text
        if "." in text[:150]:
            summary = text.split(".", 1)[0] + "."
        else:
            summary = text[:max_length]
            if len(text) > max_length:
                summary += "..."
        return summary.replace("\n", " ").strip()
    
    # Display transcripts with summaries in expander headers
    for ag in orch.agents:
        # Get latest transcript entry for summary in header
        latest = None
        if ag.transcript:
            latest = ag.transcript[-1]
        
        # Create expander header with summary if available
        if latest:
            summary = get_summary(latest["content"])
            header = f"{ag.name}: {summary}"
        else:
            header = f"{ag.name}"
            
        with st.expander(header):
            for turn in ag.transcript:
                st.markdown(f"**{turn['round'].capitalize()}** → {turn['content']}")
    # Verdicts
    for item in orch.history:
        st.info(f"Round {item['round']} Judge: {item['verdict']}")

    # Controls
    col1, col2 = st.columns(2)
    advance_key = f"advance_{orch.round_num}_{orch.phase}"  # Create unique key for each state
    
    if col1.button("Advance Round", key=advance_key) and not orch.config.auto and not orch.stopped:
        try:
            # Set button state to prevent double-clicks
            st.session_state[f"{advance_key}_clicked"] = True
            
            # Create and set the event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the next round and wait for it to complete
            loop.run_until_complete(orch.next_round(st.session_state.topic))
            
            # Close the loop properly
            loop.close()
            
            # Store the orchestrator's state in session state
            # Deep copy any necessary objects to ensure state persistence
            st.session_state.orch = orch
            st.session_state.last_update = datetime.datetime.now().isoformat()
            
            # Force Streamlit to completely rerun the app
            st.rerun()
        except Exception as e:
            st.error(f"Error advancing round: {e}")
            st.write(f"Debug - Phase: {orch.phase}, Round: {orch.round_num}")
            
    if col2.button("Stop Debate"):
        orch.stopped = True
        st.session_state.orch = orch

    st.write("\n---\n")
    st.write(f"**Phase:** {orch.phase} • **Round:** {orch.round_num}")

    # Round Counter and Winning Indicator
    st.markdown("### Debate Progress")
    col1, col2 = st.columns(2)

    # Round counter with progress bar
    total_rounds = 5  # Assume 5 rounds is a full debate
    progress = min(orch.round_num / total_rounds, 1.0)
    col1.metric("Current Round", f"{orch.round_num}")
    col1.progress(progress)

    # Winning indicator based on judge verdicts
    if orch.history:
        # Calculate scores for each agent based on judge verdicts
        agent_scores = {ag.name: 0 for ag in orch.agents}
        
        for item in orch.history:
            verdict = item['verdict']
            
            # Check if agent mentions exist in the verdict
            if 'agent_scores' in verdict:
                # Direct scores from judge
                for agent_name, score in verdict['agent_scores'].items():
                    if agent_name in agent_scores:
                        agent_scores[agent_name] += float(score)
            elif 'analysis' in verdict:
                # Parse analysis text for agent mentions
                analysis = verdict['analysis'].lower()
                for agent_name in agent_scores:
                    name_lower = agent_name.lower()
                    # Award points for positive mentions in analysis
                    if name_lower in analysis:
                        # Find nearby positive sentiment words
                        positive_words = ['strong', 'compelling', 'convincing', 'valid', 'sound', 'good']
                        for word in positive_words:
                            if word in analysis and abs(analysis.find(name_lower) - analysis.find(word)) < 50:
                                agent_scores[agent_name] += 0.5
        
        # Normalize scores between 0 and 1
        max_score = max(agent_scores.values()) if agent_scores.values() else 1
        if max_score > 0:
            normalized_scores = {name: score/max_score for name, score in agent_scores.items()}
        else:
            normalized_scores = agent_scores
        
        # Display winning indicator
        col2.subheader("Current Standing")
        
        # Sort agents by score
        sorted_agents = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create a progress bar for each agent showing their relative standing
        for name, score in sorted_agents:
            # Display color-coded progress bars (green for leading, orange for others)
            if name == sorted_agents[0][0]:  # Leading agent
                col2.markdown(f"**{name}** - Leading")
                col2.progress(score, "rgb(0, 200, 0)")
            else:
                col2.markdown(f"**{name}**")
                col2.progress(score, "rgb(255, 165, 0)")
    else:
        col2.info("Debate has just started. Standings will appear after the first round.")

    st.markdown("---")

    # Inject evidence box
    new_evidence = st.text_area("Inject new evidence (shared with all agents):", height=100)
    if st.button("Submit Evidence") and new_evidence:
        # Prepend evidence to topic for next round
        st.session_state.topic = new_evidence + "\n\n" + st.session_state.topic
        st.success("Evidence added. It will be included in next round prompts.")