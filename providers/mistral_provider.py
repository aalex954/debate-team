from __future__ import annotations
import os, httpx
from providers import Provider, register

@register("mistral")
class MistralProvider(Provider):
    _url = "https://api.mistral.ai/v1/chat/completions"
    async def complete(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY','')}"}
        json_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self._url, headers=headers, json=json_body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]