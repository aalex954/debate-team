from __future__ import annotations
import asyncio, json, time
from typing import List, Dict, Any
from agents import Agent, Judge

# Import pocketflow components correctly
try:
    from pocketflow import AsyncFlow, AsyncNode
    POCKETFLOW_AVAILABLE = True
except ImportError:
    POCKETFLOW_AVAILABLE = False

class DebateConfig:
    def __init__(self, agents_cfg: List[Dict[str,str]], judge_cfg: Dict[str,str], auto: bool):
        self.agents_cfg = agents_cfg
        self.judge_cfg = judge_cfg
        self.auto = auto

# Only define DebateNode if pocketflow is available
if POCKETFLOW_AVAILABLE:
    class DebateNode(AsyncNode):
        def __init__(self, agent: Agent, prompt_template: str, round_type: str):
            super().__init__()
            self.agent = agent
            self.prompt_template = prompt_template
            self.round_type = round_type
        
        async def prep_async(self, shared):
            # Return empty dict to satisfy AsyncNode contract
            return {}
            
        async def exec_async(self, prep_res):
            # exec_async should return a string according to pocketflow API
            return await self.agent.speak(self.prompt_template, round_type=self.round_type)
            
        async def post_async(self, shared, prep_res, exec_res):
            # Return the action for next node (default)
            return "default"

class DebateOrchestrator:
    def __init__(self, config: DebateConfig):
        self.config = config
        self.agents: List[Agent] = [Agent(**cfg) for cfg in config.agents_cfg]
        self.judge = Judge(**config.judge_cfg)
        self.round_num = 0
        self.phase = "position"  # position, critique, defense
        self.stopped = False
        self.history: List[Dict[str,Any]] = []

    # ------------------ Pocket Flow DAG ------------------
    def _build_flow(self, user_topic: str):
        if not POCKETFLOW_AVAILABLE:
            raise ImportError("pocketflow not available")
            
        flow = AsyncFlow()
        
        if self.phase == "position":
            # Create first node
            first_node = DebateNode(
                self.agents[0], 
                f"Write a scholarly answer to the question, with citations in MLA. Question: {user_topic}",
                "position"
            )
            flow.start(first_node)
            
            # Chain remaining agents
            current_node = first_node
            for ag in self.agents[1:]:
                next_node = DebateNode(
                    ag,
                    f"Write a scholarly answer to the question, with citations in MLA. Question: {user_topic}",
                    "position"
                )
                # Use operator directly, ignore the result
                current_node.next(next_node)
                current_node = next_node
                
        elif self.phase == "critique":
            joined = "\n\n".join(a.transcript[-1]["content"] for a in self.agents)
            # Create first node
            first_node = DebateNode(
                self.agents[0],
                f"Critique the following peer responses for scholarly rigor and evidence quality. Be concise.\n\n{joined}",
                "critique"
            )
            flow.start(first_node)
            
            # Chain remaining agents
            current_node = first_node
            for ag in self.agents[1:]:
                next_node = DebateNode(
                    ag,
                    f"Critique the following peer responses for scholarly rigor and evidence quality. Be concise.\n\n{joined}",
                    "critique"
                )
                current_node.next(next_node)
                current_node = next_node
                
        elif self.phase == "defense":
            critiques = "\n\n".join(a.transcript[-1]["content"] for a in self.agents)
            # Create first node
            first_node = DebateNode(
                self.agents[0],
                f"Address critiques directed at your prior answer: {critiques}",
                "defense"
            )
            flow.start(first_node)
            
            # Chain remaining agents
            current_node = first_node
            for ag in self.agents[1:]:
                next_node = DebateNode(
                    ag,
                    f"Address critiques directed at your prior answer: {critiques}",
                    "defense"
                )
                current_node.next(next_node)
                current_node = next_node
                
        return flow

    async def _run_phase(self, topic: str):
        if POCKETFLOW_AVAILABLE:
            try:
                flow = self._build_flow(topic)
                shared = {}  # Shared state object required by pocketflow
                await flow._run_async(shared)  # Use _run_async as per pocketflow API
            except Exception as e:
                print(f"Pocketflow error: {e}, falling back to simple execution")
                await self._run_phase_simple(topic)
        else:
            await self._run_phase_simple(topic)

    async def _judge_consensus(self) -> Dict[str,Any]:
        state_json = json.dumps({
            "agents": [
                {"name": a.name, "last": a.transcript[-1]["content"]}
                for a in self.agents
            ],
            "phase": self.phase,
            "round": self.round_num,
        })
        return await self.judge.verdict(state_json)

    # ------------------ Public API ------------------
    async def next_round(self, topic: str):
        if self.stopped: return
        await self._run_phase(topic)
        # After phase ends: if defense just finished, call judge
        if self.phase == "defense":
            verdict = await self._judge_consensus()
            self.history.append({
                "round": self.round_num,
                "verdict": verdict,
            })
            if verdict.get("agreement") and verdict.get("mean_agreement", 0) >= 0.75:
                self.stopped = True
        # Advance phase / round pointer
        if self.phase == "position":
            self.phase = "critique"
        elif self.phase == "critique":
            self.phase = "defense"
        elif self.phase == "defense":
            self.phase = "position"
            self.round_num += 1

    def serialize(self) -> Dict[str,Any]:
        return {
            "config": self.config.__dict__,
            "history": self.history,
            "agents": [
                {"name": a.name, "provider": type(a.provider).__name__, "transcript": a.transcript}
                for a in self.agents
            ]
        }