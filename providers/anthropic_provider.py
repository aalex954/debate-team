from __future__ import annotations
import os, httpx, asyncio
from providers import Provider, register

@register("anthropic")
class AnthropicProvider(Provider):
    _url = "https://api.anthropic.com/v1/messages"

    async def complete(self, prompt: str) -> str:
        headers = {
            "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        json_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self._url, headers=headers, json=json_body)
            r.raise_for_status()
            return r.json()["content"][0]["text"]