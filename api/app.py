"""LabMatch API.

v1 (when data/build/lambdarank.txt and doc embeddings exist): hybrid retrieval (BM25 + SPECTER2,
reciprocal rank fusion) re-ranked by the LightGBM LambdaRank model trained in ranking/experiment.py.
Falls back to the v0 TF-IDF scorer otherwise.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app = FastAPI(title="LabMatch")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.exists(os.path.join(ROOT, "data/build/lambdarank.txt")):
    from ranking.serve import Ranker
    IDX = Ranker()
else:
    from ranking.retrieve_v0 import Index
    IDX = Index()


from api.chat import make_router  # noqa: E402
app.include_router(make_router(IDX))


class Q(BaseModel):
    query: str
    k: int = 10


@app.post("/api/match")
def match(q: Q):
    return {"results": IDX.search(q.query, k=q.k)}


@app.get("/api/stats")
def stats():
    p = IDX.p
    return {"labs": int(len(p)), "funded_labs": int(((p.n_active > 0) & (p.active_amount >= 1e5)).sum()),
            "grants": int(len(IDX.aw))}
