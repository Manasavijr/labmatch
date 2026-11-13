# LabMatch

Find the lab that fits your research. Paste your interests or a paper abstract, and LabMatch ranks
US AI/ML labs by **research fit**, **active funding** and **recent output**, with the reason for each match.

It is the sequel to FundedLabs: FundedLabs answers "who is funded"; LabMatch answers "who should I work with".

## What's inside

```
pipeline/spark/bronze_grants.py     NIH + NSF extracts -> Iceberg bronze tables (PySpark, JDBC catalog)
pipeline/openalex/ingest.py         resolve grant PIs to OpenAlex authors via shared papers; pull recent works
pipeline/spark/bronze_openalex.py   OpenAlex landing JSON -> Iceberg bronze/silver (PI-author links, works)
dbt/                                staging -> intermediate -> marts (lab_profile, lab_document), with tests
airflow/dags/labmatch_dag.py        weekly DAG: ingest -> Spark -> dbt -> embed -> train/eval -> export
ranking/embed.py                    SPECTER2 embeddings of grant abstracts and paper titles
ranking/experiment.py               BM25, SPECTER2, hybrid (RRF) and LightGBM LambdaRank; held-out evaluation; MLflow
api/app.py                          FastAPI: /api/match, /api/stats
web/                                Next.js front end
artifact/                           single-file demo build of the front end
```

## Data

| Source | What | How |
|---|---|---|
| NIH RePORTER ExPORTER | projects, abstracts, PIs, grant-to-PubMed links | bulk files (cleaned by the FundedLabs pipeline) |
| NSF Award Search | awards, abstracts, PIs | bulk files |
| OpenAlex | authors, works, abstracts, topics | REST API, checkpointed and resumable |

Scope: labs (PIs) holding at least one AI/ML grant at a US institution. AI/ML is a keyword rule over grant
title and abstract (see `dbt_project.yml: ml_pattern`).

## Pipeline

1. **Bronze.** PySpark reads the raw extracts and writes Iceberg tables through a JDBC (SQLite) catalog.
   pyiceberg reads the same catalog, so dbt and Python see the same tables and snapshots.
2. **Entity resolution.** A grant PI is linked to an OpenAlex author when that author's name matches the PI on
   the PubMed papers credited to the PI's grants (vote across papers, at least 2 agreeing papers).
3. **dbt.** `lab_profile` is one row per lab: grants, active dollars, months of funding left, paper counts.
   `lab_document` holds the text units used for retrieval. Source and model tests run on every build.
4. **Retrieval.** BM25 over each lab's text and SPECTER2 cosine against each lab's embedding centroid, fused with
   reciprocal rank fusion into a top-100 candidate list.
5. **Ranking.** LightGBM LambdaRank re-ranks candidates using retrieval scores plus funding, recency and size features.

## Evaluation (no hand labels)

For every lab with at least 4 grant-linked papers, its newest paper is held out and removed from every lab's
profile. The paper's title is the query and every lab linked to that paper is relevant. Labs are split 80/20.
The ranker trains on queries from train labs and is scored on queries from test labs, reporting Recall@10/50/100,
NDCG@10 and MRR for BM25, SPECTER2, hybrid and LambdaRank. Results are written to `data/build/eval.json` and
logged to MLflow. Numbers are reported only after a run.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pipeline.spark.bronze_grants
(cd dbt && LABMATCH_ROOT=.. ../.venv/bin/dbt build --profiles-dir . --threads 1)
OPENALEX_API_KEY=... .venv/bin/python -m pipeline.openalex.ingest resolve
.venv/bin/python -m ranking.embed
.venv/bin/python -m ranking.experiment
.venv/bin/uvicorn api.app:app --port 8000 & (cd web && npm install && npm run dev)
```

Or run everything with Airflow: point `LABMATCH_ROOT` at this folder and trigger the `labmatch` DAG.

## Limits

* Grant-to-paper links come from NIH; NSF PIs rely on OpenAlex resolution for their papers.
* The AI/ML filter is keyword based, so some labs that use ML without saying so are missed.
* Held-out evaluation rewards labs that publish a lot; it measures "can we find the lab that wrote this", not
  "is this the best lab for a student".

## Help chat (Groq)

Every screen has an "Ask" helper. In the full app it calls `POST /api/chat`, which talks to Groq from the server, so the key never reaches the browser. The model can call one tool, `search_labs`, which runs the real ranker.

```bash
cp .env.example .env            # .env is git-ignored
# put your key in .env:  GROQ_API_KEY=...
uvicorn api.app:app --port 8000 # then: cd web && npm run dev
curl localhost:8000/api/chat/status
```

Without a key the endpoint returns 503 and the widget falls back to built-in answers. The published single-file demo can't reach a server, so it uses Claude through the page, with the same FAQ fallback.
