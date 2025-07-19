import asyncio, json, datetime, os, time  # Add time module here
import streamlit as st
from orchestrator import DebateConfig, DebateOrchestrator
from storage import save_session, load_session

st.set_page_config(page_title="Scholarly Agent Debate", layout="wide")

# ------------- Sidebar settings ----------------
# Replace the header with an expander
with st.sidebar.expander("Agent Config", expanded=True):
    def default_agents():
        return [
            {"name": "OpenAI", "provider_name": "openai", "model": "gpt-4o-mini"},
            {"name": "Mistral", "provider_name": "mistral", "model": "mistral-large-latest"},
        ]

    # Initialize agent configs if not in session state
    if "agent_cfgs" not in st.session_state:
        st.session_state.agent_cfgs = default_agents()
        st.session_state.prev_a_num = len(st.session_state.agent_cfgs)
    
    # Number input for agent count
    a_num = st.number_input("# Agents", min_value=2, max_value=8, value=len(st.session_state.agent_cfgs))
    
    # Handle changes in agent count
    if a_num != st.session_state.prev_a_num:
        # Adding new agents
        if a_num > st.session_state.prev_a_num:
            # Add new default agents with unique names
            for i in range(st.session_state.prev_a_num, a_num):
                if i % 2 == 0:  # Alternate between providers for diversity
                    new_agent = {"name": f"OpenAI-{i+1}", "provider_name": "openai", "model": "gpt-4o-mini"}
                else:
                    new_agent = {"name": f"Mistral-{i+1}", "provider_name": "mistral", "model": "mistral-large-latest"}
                st.session_state.agent_cfgs.append(new_agent)
        # Removing agents
        else:
            st.session_state.agent_cfgs = st.session_state.agent_cfgs[:a_num]
        
        # Update previous count
        st.session_state.prev_a_num = a_num

    # Display configuration controls for each agent
    for i in range(a_num):
        cfg = st.session_state.agent_cfgs[i]
        st.text_input(f"Agent {i+1} Name", value=cfg["name"], key=f"name{i}")
        provider_options = ["openai", "anthropic", "mistral", "local"]
        st.selectbox(
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
        st.selectbox(
            "Model", model_options.get(selected_provider, [cfg["model"]]),
            index=model_options.get(selected_provider, [cfg["model"]]).index(cfg["model"]) if cfg["model"] in model_options.get(selected_provider, [cfg["model"]]) else 0,
            key=f"model{i}")

    judge_model = st.text_input("Judge Model (OpenAI)", value="gpt-4o-mini")
    auto_run = st.checkbox("Auto‑advance rounds", value=False)

# Keep these sections outside the expander
st.sidebar.markdown("---")
if st.sidebar.button("Save Session") and "orch" in st.session_state:
    path = f"debate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_session(path, st.session_state.orch)
    st.sidebar.success(f"Saved → {path}")

# Always show file uploader (not conditional on button click)
uploaded = st.sidebar.file_uploader("Load Debate Session", type="json", key="debate_file")

# Add a flag in session state to track if we've processed this file already
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# Add a separate load button that checks if a file is uploaded
if uploaded and st.sidebar.button("Load Selected File"):
    # Create a file identifier using name and size
    file_identifier = f"{uploaded.name}_{uploaded.size}"
    
    # Check if we've already processed this file
    if file_identifier != st.session_state.last_processed_file:
        try:
            # Read the data
            data = json.load(uploaded)
            
            # Debug info to verify data structure
            st.sidebar.write(f"Found {len(data['agents'])} agents in file")
            
            # Reconstruct the DebateConfig from saved data
            agents_cfg = data["config"]["agents_cfg"]
            judge_cfg = data["config"]["judge_cfg"]
            auto = data["config"]["auto"]
            config = DebateConfig(agents_cfg, judge_cfg, auto)
            
            # Create a new orchestrator with the config
            orch = DebateOrchestrator(config)
            
            # Restore orchestrator state
            orch.history = data["history"]
            orch.round_num = data["history"][-1]["round"] if data["history"] else 0
            
            # Determine current phase based on last agent transcript entry
            if data["agents"] and data["agents"][0]["transcript"]:
                last_round_type = data["agents"][0]["transcript"][-1]["round"]
                if last_round_type == "position":
                    orch.phase = "critique"
                elif last_round_type == "critique":
                    orch.phase = "defense"
                elif last_round_type == "defense":
                    orch.phase = "position"
                    orch.round_num += 1
            
            # Restore agent transcripts
            for i, agent_data in enumerate(data["agents"]):
                if i < len(orch.agents):  # Make sure we don't go out of bounds
                    orch.agents[i].transcript = agent_data["transcript"]
            
            # Save to session state
            st.session_state.orch = orch
            
            # Try to extract original topic from position statements if available
            if data["agents"] and data["agents"][0]["transcript"]:
                for entry in data["agents"][0]["transcript"]:
                    if entry["round"] == "position":
                        # Get the first few words of the position statement as topic
                        content = entry["content"]
                        topic_hint = content.split("Question:", 1)[-1].strip()
                        if topic_hint:
                            st.session_state.topic = topic_hint[:100] + "..."
                            break
                
            # If we couldn't extract a topic, use a placeholder
            if "topic" not in st.session_state or not st.session_state.topic:
                st.session_state.topic = "Loaded debate session"
            
            # Mark this file as processed
            st.session_state.last_processed_file = file_identifier
            
            st.sidebar.success("Session loaded successfully!")
            st.rerun()  # Use rerun instead of experimental_rerun
        except Exception as e:
            st.sidebar.error(f"Error loading session: {e}")
            st.sidebar.write(f"Error details: {type(e).__name__}: {str(e)}")

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
    
    # Auto-advance functionality
    if orch.config.auto and not orch.stopped:
        # Check if we need to create a new task
        if "auto_advance_task" not in st.session_state or st.session_state.get("auto_advance_status") != "running":
            st.session_state.auto_advance_status = "running"
            
            # Create a placeholder to show auto-advance status
            auto_status = st.empty()
            auto_status.info("Auto-advancing rounds enabled")
            
            # Only advance if not at the end of a round or stopped
            if not orch.stopped:
                try:
                    # Create and set the event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Run the next round
                    loop.run_until_complete(orch.next_round(st.session_state.topic))
                    
                    # Close the loop properly
                    loop.close()
                    
                    # Update session state
                    st.session_state.orch = orch
                    st.session_state.last_update = datetime.datetime.now().isoformat()
                    
                    # Set a short delay before refreshing to avoid UI flashing
                    time.sleep(1)
                    
                    # Rerun the app to show the updated state
                    st.rerun()
                except Exception as e:
                    st.error(f"Auto-advance error: {e}")
                    st.session_state.auto_advance_status = "error"

    if col1.button("Advance Round", key=advance_key) and not orch.stopped:
        # If auto is enabled, just show a message
        if orch.config.auto:
            st.info("Auto-advance is enabled. Rounds will progress automatically.")
        else:
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