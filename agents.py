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
        prompt = (
            "You are a scholarly debate judge. Given the JSON debate state below, "
            "decide if agents have reached substantial agreement. Return JSON: "
            "{\"agreement\":bool,\"mean_agreement\":float,\"explanation\":str}.\n\n" + debate_state_json
        )
        raw = await self.provider.complete(prompt)
        try:
            return json.loads(re.search(r"\{.*\}", raw, re.S).group())
        except Exception:
            return {"agreement": False, "mean_agreement": 0, "explanation": raw[:200]}