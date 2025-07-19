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
        prompt = f"""
🎓  You are the sole adjudicator of a three‑phase scholarly debate
      (POSITION → CRITIQUE → DEFENSE).

Your job:  
1. Determine whether the agents have reached "substantial agreement"  
   • Two positions are "in agreement" when their core theses are semantically ≥ 0.75 similar
     **or** when one agent clearly concedes to another in DEFENSE.  
   • Compute mean_agreement = average pairwise agreement across all agents (0‑1).  

2. Audit DEFENSE compliance for **each agent** against ALL FOUR rules.  
   • **Concede‑or‑Counter rule** — Exactly one of: a ≤ 10‑word concession **or** a ≤ 100‑word rebuttal.  
   • **One fresh citation** — At most ONE new MLA citation appears.  
   • **Updated Fragility Index** — Index is present, changed only when justified, and the change is explained in ≤ 1 sentence.  
   • **Roadmap to definitive proof** — Concludes with *exactly* 2 sentences describing decisive future evidence.  

   For any violation, subtract 0.10 from that agent's agreement score (but do not go below 0).  

3. Return a JSON object with **exactly** these keys:  
   {{
     "agreement": <bool ‑‑ True if mean_agreement ≥ 0.75 *after* any penalties>,  
     "mean_agreement": <float 0‑1 rounded to 2 dp>,  
     "explanation": "<concise reasoning (≤ 75 words) explaining the decision>"
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
                return {"agreement": False, "mean_agreement": 0, "explanation": raw[:200]}
        except Exception:
            return {"agreement": False, "mean_agreement": 0, "explanation": raw[:200]}