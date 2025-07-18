from __future__ import annotations
import os, asyncio
import httpx, json
from providers import Provider, register

@register("openai")
class OpenAIProvider(Provider):
    _url = "https://api.openai.com/v1/chat/completions"

    async def complete(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}",
        }
        json_body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self._url, headers=headers, json=json_body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]