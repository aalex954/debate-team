from __future__ import annotations
import asyncio, re, json, uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from providers import create as create_provider

_embedder = SentenceTransformer("all-mpnet-base-v2")

class Agent:
    def __init__(self, name: str, provider_name: str, model: str):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.provider = create_provider(provider_name, model)
        self.transcript: List[Dict[str, Any]] = []  # list of dicts per turn

    async def speak(self, prompt: str, round_type: str) -> str:
        reply = await self.provider.complete(prompt)
        self.transcript.append({"round": round_type, "content": reply})
        return reply

    def similarity(self, other_content: str) -> float:
        emb1 = _embedder.encode(self.transcript[-1]["content"], convert_to_tensor=True)
        emb2 = _embedder.encode(other_content, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

class Judge(Agent):
    """Special agent that receives whole debate and returns verdict JSON."""
    async def verdict(self, debate_state_json: str) -> Dict[str, Any]:
        debate_type = json.loads(debate_state_json).get("config", {}).get("debate_type", "non-binary")
        
        if debate_type == "binary":
            prompt = f"""
🎓 You are the sole adjudicator of a factual debate where the goal is objective correctness.

Your job:  
1. Determine which position is most factually accurate and well-supported by evidence  
   • Evaluate each position primarily on: factual accuracy, quality of citations, logical consistency
   • Assign a correctness score (0-1) to each agent based on these criteria

2. Audit DEFENSE compliance for **each agent** against ALL FOUR rules.  
   • **Concede-or-Counter rule** — Exactly one of: a ≤ 10-word concession **or** a ≤ 100-word rebuttal.  
   • **One fresh citation** — At most ONE new MLA citation appears.  
   • **Updated Fragility Index** — Index is present, changed only when justified, and the change is explained in ≤ 1 sentence.  
   • **Roadmap to definitive proof** — Concludes with *exactly* 2 sentences describing decisive future evidence.  

   For any violation, subtract 0.10 from that agent's correctness score (but do not go below 0).  

3. Return a JSON object with **exactly** these keys:  
   {{
     "most_correct_agent": "<name of agent with highest correctness score>",
     "correctness_scores": {{<agent_name>: <score 0-1 for each agent>}},
     "key_facts": ["<list of 3-5 key factual points established in the debate>"],
     "explanation": "<concise reasoning (≤ 75 words) explaining your decision>"
   }}

DEBATE_STATE_JSON:
{debate_state_json}
"""
        else:  # non-binary
            prompt = f"""
🎓 You are the sole adjudicator of an exploratory debate where the goal is ideation and topic exploration.

Your job:  
1. Evaluate the quality of exploration and ideation  
   • Assess each position on: novelty of ideas, breadth of perspectives, insightful connections
   • Assign an exploration score (0-1) to each agent based on these criteria

2. Identify key insights and novel perspectives that emerged during the debate
   • Extract 3-5 most interesting or valuable ideas from the entire debate
   • Note any unexpected connections or synthesis between initially different positions

3. Return a JSON object with **exactly** these keys:  
   {{
     "most_insightful_agent": "<name of agent with highest exploration score>",
     "exploration_scores": {{<agent_name>: <score 0-1 for each agent>}},
     "key_insights": ["<list of 3-5 most valuable ideas from the debate>"],
     "novel_connections": ["<list of unexpected connections or synthesis points>"],
     "explanation": "<concise reasoning (≤ 75 words) on the value of the exploration>"
   }}

DEBATE_STATE_JSON:
{debate_state_json}
"""
        
        raw = await self.provider.complete(prompt)
        try:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                return json.loads(match.group())
            else:
                return {"explanation": raw[:200]}
        except Exception:
            return {"explanation": raw[:200]}