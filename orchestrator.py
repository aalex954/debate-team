from __future__ import annotations
import asyncio, json, time
from typing import List, Dict, Any
from agents import Agent, Judge
from pocket_flow import Flow, Node   # lightweight async flow engine

class DebateConfig(BaseException):
    def __init__(self, agents_cfg: List[Dict[str,str]], judge_cfg: Dict[str,str], auto: bool):
        self.agents_cfg = agents_cfg
        self.judge_cfg = judge_cfg
        self.auto = auto

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
    def _build_flow(self, user_topic: str) -> Flow:
        flow = Flow(name="debate_round")
        # Nodes per agent depending on phase
        if self.phase == "position":
            for ag in self.agents:
                flow.add(Node(
                    id=f"pos-{ag.id}",
                    coro=lambda a=ag: a.speak(
                        f"Write a scholarly answer to the question, with citations in MLA. Question: {user_topic}",
                        round_type="position"
                    )
                ))
        elif self.phase == "critique":
            joined = "\n\n".join(a.transcript[-1]["content"] for a in self.agents)
            for ag in self.agents:
                flow.add(Node(
                    id=f"crit-{ag.id}",
                    coro=lambda a=ag, j=john:=joined: a.speak(
                        f"Critique the following peer responses for scholarly rigor and evidence quality. Be concise.\n\n{j}",
                        round_type="critique"
                    )
                ))
        elif self.phase == "defense":
            critiques = "\n\n".join(a.transcript[-1]["content"] for a in self.agents)
            for ag in self.agents:
                flow.add(Node(
                    id=f"def-{ag.id}",
                    coro=lambda a=ag, c=critiques: a.speak(
                        f"Address critiques directed at your prior answer: {c}",
                        round_type="defense"
                    )
                ))
        return flow

    async def _run_phase(self, topic: str):
        flow = self._build_flow(topic)
        await flow.run_async()
        # Collect latest outputs already stored inside agents' transcripts

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
            if verdict["agreement"] and verdict["mean_agreement"] >= 0.75:
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