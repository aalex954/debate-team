"""Provider registry + base classes."""
from __future__ import annotations
import abc, os, asyncio, json
from typing import Any, Dict
import httpx

class Provider(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    async def complete(self, prompt: str) -> str: ...

# ---------------- Registry ➜ name→cls map ---------------
_REG: Dict[str, type[Provider]] = {}

def register(name: str):
    def _wrap(cls):
        _REG[name] = cls
        return cls
    return _wrap

def create(name: str, model: str) -> Provider:
    if name not in _REG:
        raise ValueError(f"Unknown provider '{name}'. Registered: {list(_REG)}")
    return _REG[name](model)
