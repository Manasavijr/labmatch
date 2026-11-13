"""Help chat for LabMatch, answered by Groq on the server.

The key never reaches the browser: it is read from the GROQ_API_KEY environment variable
(or a git-ignored .env file in the repo root). The model can call one tool, search_labs,
which runs the real ranker, so answers about "which labs work on X" come from LabMatch data.
"""
import json
import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _load_dotenv(path=os.path.join(ROOT, ".env")):
    """Tiny .env reader so the repo needs no extra dependency. Real env vars win."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

SYSTEM = """You are the built-in helper inside LabMatch, a web app that helps students and researchers find a US AI/ML research lab.
Answer in 1-3 short, friendly sentences of plain language for someone who may not know machine learning.
Only state facts given here or returned by tools. If you are unsure, say so.
About LabMatch: the user describes their research; LabMatch ranks US AI/ML labs that hold active NIH or NSF research grants.
Data: NIH RePORTER, NSF Awards, OpenAlex papers, built by a pipeline of Airflow, PySpark, Apache Iceberg and dbt.
Ranking: BM25 keyword search and SPECTER2 embeddings are fused (reciprocal rank fusion), then re-ranked by a LightGBM LambdaRank model with funding and recency features.
Scores are shown relative to the top match (100); the calibrated odds are in the score tooltip.
Screens: Overview, Match (search, ranked list, reasons), Landscape (map of labs by topic), Lab (one lab's funding, grants, papers), Shortlist (saved labs, stored only in the browser), How it works.
Use search_labs whenever the user asks which labs work on a topic. LabMatch was built by Manasa Vijayaraghavan."""

TOOLS = [{"type": "function", "function": {
    "name": "search_labs",
    "description": "Find the best-matching funded US AI/ML labs for a research topic. Returns up to 5 labs.",
    "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}}]


class Msg(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=20)
    context: str = Field("", max_length=4000)  # what the user is looking at: screen, query, top matches


def make_router(index):
    router = APIRouter()

    def search_labs(topic: str):
        return [{"name": r["name"], "institution": r["institution"], "state": r["state"],
                 "active_funding_usd": round(r["active_amount"]), "funded_until": r["funded_until"],
                 "matched_on": r["why"]["terms"]} for r in index.search(topic, k=5)]

    @router.get("/api/chat/status")
    def status():
        return {"enabled": bool(os.environ.get("GROQ_API_KEY")), "model": MODEL}

    @router.post("/api/chat")
    def chat(body: ChatIn):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise HTTPException(503, "Chat is off: set GROQ_API_KEY on the server.")
        msgs = [{"role": "system", "content": SYSTEM + ("\n\nWhat the user is looking at:\n" + body.context if body.context else "")}]
        msgs += [{"role": m.role if m.role in ("user", "assistant") else "user", "content": m.content} for m in body.messages[-10:]]
        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=30) as client:
            for _ in range(3):  # at most two tool rounds, then a final answer
                r = client.post(GROQ_URL, headers=headers, json={"model": MODEL, "messages": msgs, "tools": TOOLS,
                                                                  "temperature": 0.3, "max_tokens": 400})
                if r.status_code != 200:
                    raise HTTPException(502, f"Groq error {r.status_code}")
                m = r.json()["choices"][0]["message"]
                calls = m.get("tool_calls") or []
                if not calls:
                    return {"answer": (m.get("content") or "").strip(), "model": MODEL}
                msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
                for c in calls:
                    try:
                        out = search_labs(json.loads(c["function"]["arguments"]).get("topic", ""))
                    except Exception as e:  # bad arguments from the model: tell it, don't crash
                        out = {"error": str(e)}
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
        return {"answer": "Sorry, I couldn't finish that one. Try asking a bit differently.", "model": MODEL}

    return router
