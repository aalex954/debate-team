import asyncio, json, datetime, os, time  # Add time module here
import streamlit as st
from orchestrator import DebateConfig, DebateOrchestrator
from storage import save_session, load_session

st.set_page_config(page_title="Multi Agentic System Debate", layout="wide")

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

# Add this function before it's used in the UI controls section

def start_debate(topic):
    """Initialize a new debate with the given topic."""
    # Build config from sidebar widgets
    cfgs = []
    for i in range(a_num):
        cfgs.append({
            "name": st.session_state[f"name{i}"],
            "provider_name": st.session_state[f"prov{i}"],
            "model": st.session_state[f"model{i}"],
        })
    judge_cfg = {"name": "Judge", "provider_name": "openai", "model": judge_model}
    
    # Get auto-advance setting
    auto_run = st.session_state.get("auto_advance", False)
    
    # Create debate config and orchestrator
    conf = DebateConfig(cfgs, judge_cfg, auto_run)
    st.session_state.orch = DebateOrchestrator(conf)
    st.session_state.topic = topic
    
    # Run the first round asynchronously
    try:
        # Create and set the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the first round and wait for it to complete
        loop.run_until_complete(st.session_state.orch.next_round(topic))
        
        # Close the loop properly
        loop.close()
        
        # Set timestamp for last update
        st.session_state.last_update = datetime.datetime.now().isoformat()
        
        # Rerun to refresh UI
        st.rerun()
    except Exception as e:
        st.error(f"Error starting debate: {e}")

# Replace the existing UI control section with this consolidated implementation

# ------------- Main UI ----------------
st.title("📚 Multi Agentic System Debate")

# Create consistent layout for topic and controls
topic_col, button_col, toggle_col = st.columns([5, 2, 2])

# Topic input/display (always in the same position)
with topic_col:
    if "topic" in st.session_state and "orch" in st.session_state:
        st.markdown(f"**Topic:** {st.session_state.topic}")
    else:
        user_topic = st.text_input("Enter a topic or question:", key="topic_input")
        if user_topic:
            st.session_state.topic = user_topic

# Determine app state
debate_in_progress = "orch" in st.session_state
is_auto_mode = debate_in_progress and st.session_state.orch.config.auto if debate_in_progress else False
is_stopped = debate_in_progress and st.session_state.orch.stopped if debate_in_progress else False

# Auto-advance toggle (always in the same position)
with toggle_col:
    # In non-debate state, show disabled toggle
    if not debate_in_progress:
        st.toggle("Auto-advance", value=False, disabled=True, key="auto_toggle_init")
    else:
        # Initialize auto_advance in session state if needed
        if "auto_advance" not in st.session_state:
            st.session_state.auto_advance = st.session_state.orch.config.auto
        
        # Create toggle with appropriate styling
        auto_toggle = st.toggle(
            "Auto-advance", 
            value=st.session_state.auto_advance,
            key="auto_advance_toggle",
            disabled=is_stopped
        )
        
        # Handle toggle change
        if auto_toggle != st.session_state.orch.config.auto:
            st.session_state.orch.config.auto = auto_toggle
            st.session_state.auto_advance = auto_toggle
            st.rerun()

# Main action button (always in the same position)
with button_col:
    if not debate_in_progress:
        # Start debate button
        start_disabled = not st.session_state.get("topic", "")
        if st.button("Start Debate", type="primary", disabled=start_disabled, key="main_action"):
            start_debate(st.session_state.topic)
    else:
        if not is_stopped:
            # Stop debate button
            if st.button("Stop Debate", type="secondary", key="main_action"):
                st.session_state.orch.stopped = True
                st.session_state.orch.config.auto = False
                st.session_state.auto_advance = False
                st.rerun()
        else:
            # New debate button
            if st.button("New Debate", type="primary", key="main_action"):
                # Clear the orchestrator to start fresh
                if "orch" in st.session_state:
                    del st.session_state.orch
                if "topic" in st.session_state:
                    del st.session_state.topic
                if "auto_advance" in st.session_state:
                    del st.session_state.auto_advance
                st.rerun()

# Advance Round button (only appears in manual mode, but in consistent position)
advance_col = st.container()
with advance_col:
    if debate_in_progress and not is_auto_mode and not is_stopped:
        orch = st.session_state.orch
        advance_key = f"advance_{orch.round_num}_{orch.phase}"
        if st.button("Advance Round", key=advance_key, type="primary"):
            try:
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
    else:
        # Empty placeholder to maintain layout
        st.markdown("&nbsp;", unsafe_allow_html=True)

# Process auto-advance if enabled
if debate_in_progress and is_auto_mode and not is_stopped:
    orch = st.session_state.orch
    
    # Create a placeholder for auto-advance status
    auto_status = st.empty()
    auto_status.info("Auto-advancing rounds...")
    
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
        
        # Add a short delay to prevent UI flashing too quickly
        time.sleep(1)
        
        # Rerun to continue auto-advancing
        st.rerun()
    except Exception as e:
        st.error(f"Auto-advance error: {e}")
        # Turn off auto but don't stop the debate
        orch.config.auto = False
        st.session_state.auto_advance = False
        st.session_state.orch = orch

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
    
    # Generate consistent colors for each agent
    import colorsys
    
    # Helper for creating agent colors
    def get_agent_colors(num_agents):
        colors = []
        for i in range(num_agents):
            # Generate evenly spaced hues
            hue = i / num_agents
            # Convert HSV to RGB (using saturation=0.7, value=0.9 for readability)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            # Format as RGB string for Streamlit
            colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
        return colors
    
    agent_colors = get_agent_colors(len(orch.agents))
    judge_color = "rgb(200, 200, 220)"  # Light blue-gray for judge
    
    # Create tabs for different views
    timeline_tab, current_tab, full_tab = st.tabs(["Timeline", "Current Round", "Full Transcript"])
    
    with timeline_tab:
        st.subheader("Debate Timeline")
        
        # Timeline view - horizontal timeline with phases
        phases = ["position", "critique", "defense"]
        max_round = orch.round_num
        
        # Track clicked elements in session state
        if "selected_turn" not in st.session_state:
            st.session_state.selected_turn = None
        
        # Create timeline
        for round_num in range(max_round + 1):
            st.markdown(f"### Round {round_num}")
            
            # Create columns for each phase
            cols = st.columns(len(phases))
            
            for i, phase in enumerate(phases):
                with cols[i]:
                    st.markdown(f"**{phase.capitalize()}**")
                    
                    # Display agent responses for this phase and round
                    for agent_idx, agent in enumerate(orch.agents):
                        color = agent_colors[agent_idx]
                        
                        # Find matching transcript entry
                        matching_entries = [t for t in agent.transcript 
                                           if t["round"] == phase and 
                                              (round_num == 0 or round_num == orch.round_num)]
                        
                        if matching_entries:
                            entry = matching_entries[-1]  # Get the latest matching entry
                            
                            # Create a unique key for this turn
                            turn_key = f"{agent.name}_{round_num}_{phase}"
                            
                            # Create a styled button with agent's specific color
                            button_html = f"""
<div style="
    background-color: {color}; 
    color: white; 
    padding: 8px 12px;
    border-radius: 4px;
    text-align: center;
    margin: 4px 0px;
    cursor: pointer;
    font-weight: bold;
    width: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);" 
    onclick="document.dispatchEvent(new CustomEvent('streamlit:buttonClicked', {{detail: '{turn_key}'}}))">
    {agent.name}
</div>
"""
                            st.markdown(button_html, unsafe_allow_html=True)

                            if turn_key in st.session_state and st.session_state[turn_key]:
                                st.session_state.selected_turn = (agent.name, round_num, phase, entry["content"])
                                # Reset the button state
                                st.session_state[turn_key] = False
        
        # Display selected turn content
        if st.session_state.selected_turn:
            agent_name, round_num, phase, content = st.session_state.selected_turn
            st.markdown("---")
            st.markdown(f"### {agent_name} - Round {round_num} ({phase.capitalize()})")
            st.markdown(content)
            if st.button("Clear Selection", key="clear_selection"):
                st.session_state.selected_turn = None
        
        # Display judge verdicts in timeline
        st.markdown("### Judge Verdicts")
        for item in orch.history:
            round_num = item['round']
            verdict = item['verdict']
            
            # Format verdict nicely
            if isinstance(verdict, dict):
                explanation = verdict.get('explanation', 'No explanation provided')
                agreement = verdict.get('agreement', False)
                st.markdown(
                    f"""<div style="padding:10px; border-left:5px solid {judge_color}; 
                    background-color:#2E3C50; color: white; border-radius: 4px;">
                    <strong>Round {round_num} Verdict:</strong> {"Agreement" if agreement else "No agreement yet"}<br>
                    {explanation}</div>""", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="padding:10px; border-left:5px solid {judge_color}; 
                    background-color:#2E3C50; color: white; border-radius: 4px;">
                    <strong>Round {round_num} Verdict:</strong> {verdict}</div>""", 
                    unsafe_allow_html=True
                )
    
    with current_tab:
        st.subheader("Current Round")
        
        # Show compact view of current round responses
        for agent_idx, agent in enumerate(orch.agents):
            if agent.transcript:
                latest = agent.transcript[-1]
                color = agent_colors[agent_idx]
                
                # Create styled container for each agent's latest response
                st.markdown(
                    f"""<div style="padding:10px; border-left:5px solid {color}; margin-bottom:10px;">
                    <strong style="color:{color}">{agent.name}</strong> ({latest['round'].capitalize()})
                    </div>""", 
                    unsafe_allow_html=True
                )
                
                with st.expander("View response"):
                    st.markdown(latest["content"])
        
        # Show latest judge verdict if available
        if orch.history:
            latest_verdict = orch.history[-1]['verdict']
            st.markdown(
                f"""<div style="padding:10px; border-left:5px solid {judge_color}; 
                background-color:#2E3C50; color: white; border-radius: 4px; margin-top:20px;">
                <strong>Judge Verdict:</strong></div>""", 
                unsafe_allow_html=True
            )
            with st.expander("View verdict"):
                st.json(latest_verdict)
    
    with full_tab:
        st.subheader("Full Transcript")
        
        # Create tabs for each agent
        agent_tabs = st.tabs([agent.name for agent in orch.agents] + ["Judge"])
        
        # Display full transcript for each agent
        for agent_idx, tab in enumerate(agent_tabs[:-1]):  # All except Judge tab
            with tab:
                agent = orch.agents[agent_idx]
                color = agent_colors[agent_idx]
                
                for turn in agent.transcript:
                    # Create timestamp
                    st.markdown(
                        f"""<div style="padding:5px; border-left:5px solid {color}; margin-bottom:5px;">
                        <strong>Round {turn.get('round_num', 0)} - {turn['round'].capitalize()}</strong>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    st.markdown(turn["content"])
                    st.markdown("---")
        
        # Display judge verdicts in the Judge tab
        with agent_tabs[-1]:  # Judge tab
            for item in orch.history:
                st.markdown(
                    f"""<div style="padding:5px; border-left:5px solid {judge_color}; 
                    background-color:#2E3C50; color: white; border-radius: 4px; margin-bottom:5px;">
                    <strong>Round {item['round']} Verdict</strong>
                    </div>""", 
                    unsafe_allow_html=True
                )
                st.json(item['verdict'])
                st.markdown("---")

    # Round Counter and Winning Indicator - Keep this section but make more compact
    col1, col2 = st.columns(2)
    with col1:
        # Round counter with progress bar
        total_rounds = 5  # Assume 5 rounds is a full debate
        progress = min(orch.round_num / total_rounds, 1.0)
        st.markdown(f"**Round {orch.round_num}/{total_rounds}** • Phase: {orch.phase.capitalize()}")
        st.progress(progress)

    with col2:
        # Winning indicator based on judge verdicts
        if orch.history:
            # Calculate scores for each agent based on judge verdicts
            agent_scores = {ag.name: 0 for ag in orch.agents}
            
            for item in orch.history:
                verdict = item['verdict']
                
                # Check if agent mentions exist in the verdict
                if isinstance(verdict, dict):
                    if 'agent_scores' in verdict:
                        for agent_name, score in verdict['agent_scores'].items():
                            if agent_name in agent_scores:
                                agent_scores[agent_name] += float(score)
                    elif 'explanation' in verdict:
                        analysis = verdict['explanation'].lower()
                        for agent_name in agent_scores:
                            name_lower = agent_name.lower()
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
            
            # Display compact standings
            sorted_agents = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (name, score) in enumerate(sorted_agents):
                # Get the agent's color
                agent_idx = next((idx for idx, ag in enumerate(orch.agents) if ag.name == name), 0)
                color = agent_colors[agent_idx]
                
                # Create a compact representation
                leader_tag = " 🏆 Leading" if i == 0 else ""
                st.markdown(f"**{name}**{leader_tag}")
                st.progress(score, color)
        else:
            st.info("Waiting for judge verdict...")

    st.markdown("---")

    # Inject evidence box
    new_evidence = st.text_area("Inject new evidence (shared with all agents):", height=100)
    if st.button("Submit Evidence") and new_evidence:
        # Prepend evidence to topic for next round
        st.session_state.topic = new_evidence + "\n\n" + st.session_state.topic
        st.success("Evidence added. It will be included in next round prompts.")