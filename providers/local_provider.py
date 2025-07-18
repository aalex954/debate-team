"""Local provider via Ollama REST API on http://localhost:11434/api/chat"""
from __future__ import annotations
import os, httpx, asyncio, json
from providers import Provider, register

@register("local")
class LocalProvider(Provider):
    _url = "http://localhost:11434/api/chat"
    async def complete(self, prompt: str) -> str:
        json_body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self._url, json=json_body)
            r.raise_for_status()
            res = r.json()
            if "message" in res:
                return res["message"]["content"]
            return res.get("response", "")