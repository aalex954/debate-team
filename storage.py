import json, datetime, os

def save_session(path: str, orchestrator):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(orchestrator.serialize(), f, indent=2)

def load_session(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
