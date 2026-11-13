"""OpenAlex ingest: resolve grant PIs to OpenAlex authors, then pull their recent works.

Step 1  works-by-PMID   For each lab, take PMIDs of papers credited to its NIH grants and fetch
                        those works (50 PMIDs per call). The authorship whose name matches the PI
                        (last name + first initial) gives the PI's OpenAlex author id and institution.
                        A PI is linked when the same author id wins on >= 2 papers (or the only paper).
Step 2  author works    For linked authors, fetch works since FROM_YEAR (title, abstract, topics,
                        authorships), 50 author ids per filter, cursor-paged.

Raw responses land as gzipped JSON lines under data/lake/landing/openalex/<step>/<run_ts>/ and are
parsed into Iceberg by pipeline/spark/bronze_openalex.py. Set OPENALEX_API_KEY for the keyed quota;
without it the shared anonymous budget applies. The client stops cleanly when the budget runs out
and the next run resumes from the last finished batch (checkpoint file per step).
"""
import gzip, json, os, re, sys, time, unicodedata
import duckdb, httpx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAND = os.path.join(ROOT, "data", "lake", "landing", "openalex")
API = "https://api.openalex.org"
KEY = os.environ.get("OPENALEX_API_KEY")
FROM_YEAR = int(os.environ.get("OPENALEX_FROM_YEAR", "2019"))
WORK_FIELDS = "id,doi,ids,title,publication_year,abstract_inverted_index,primary_topic,authorships,cited_by_count"


class BudgetExhausted(Exception):
    pass


def client():
    return httpx.Client(base_url=API, timeout=60, headers={"User-Agent": "labmatch/0.1"})


def get(c, path, params):
    if KEY:
        params = {**params, "api_key": KEY}
    for attempt in range(6):
        r = c.get(path, params=params)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 and "budget" in r.text.lower():
            raise BudgetExhausted(r.headers.get("retry-after", "?"))
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt); continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {path} {params}")


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


def name_match(pi_last, pi_first, display_name):
    t = norm(display_name)
    return bool(t) and t[-1] == pi_last and bool(pi_first) and t[0][:1] == pi_first[:1]


def checkpoint(step):
    p = os.path.join(LAND, step, "_checkpoint.json")
    return p, (json.load(open(p)) if os.path.exists(p) else {"done": []})


def write_batch(step, run, n, records):
    d = os.path.join(LAND, step, run); os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, f"part-{n:05d}.jsonl.gz"), "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def step_resolve(limit_labs=None):
    con = duckdb.connect(os.path.join(ROOT, "data/lake/labmatch.duckdb"), read_only=True)
    rows = con.sql("""
        select lp.inv_id, inv.last_name, inv.first_name, list(distinct p.pmid) as pmids
        from lab_profile lp
        join main.stg_investigator inv using (inv_id)
        join int_lab_publication p using (inv_id)
        where p.pmid is not null and lp.n_active > 0
        group by 1, 2, 3 order by max(lp.active_amount) desc
    """).fetchall()
    if limit_labs: rows = rows[:limit_labs]
    ck_path, ck = checkpoint("resolve"); done = set(ck["done"])
    run = time.strftime("%Y%m%dT%H%M%S"); c = client(); n = 0
    try:
        for inv_id, last, first, pmids in rows:
            if inv_id in done: continue
            pmids = sorted(pmids, reverse=True)[:50]  # most recent PMIDs first
            res = get(c, "/works", {"filter": "pmid:" + "|".join(map(str, pmids)), "per-page": 50,
                                    "select": "id,ids,publication_year,authorships"})
            votes = {}
            for w in res.get("results", []):
                for a in w.get("authorships", []):
                    au = a.get("author") or {}
                    if name_match(last, first, au.get("display_name")):
                        k = au.get("id"); votes.setdefault(k, {"n": 0, "name": au.get("display_name"), "orcid": au.get("orcid"),
                                                              "institutions": a.get("institutions", [])})
                        votes[k]["n"] += 1
            write_batch("resolve", run, n, [{"inv_id": inv_id, "n_works": len(res.get("results", [])), "votes": votes}])
            n += 1; done.add(inv_id)
            if n % 50 == 0:
                json.dump({"done": sorted(done)}, open(ck_path, "w")); print("resolved", n, flush=True)
    except BudgetExhausted as e:
        print(f"OpenAlex budget exhausted after {n} labs; resets in {e}s. Re-run to resume.", flush=True)
    json.dump({"done": sorted(done)}, open(ck_path, "w"))
    return n


def step_works(author_ids):
    ck_path, ck = checkpoint("works"); done = set(ck["done"])
    todo = [a for a in author_ids if a not in done]
    run = time.strftime("%Y%m%dT%H%M%S"); c = client(); n = 0
    try:
        for s in range(0, len(todo), 50):
            batch = todo[s:s + 50]; cursor = "*"; recs = []
            ids = "|".join(a.rsplit("/", 1)[-1] for a in batch)
            while cursor:
                res = get(c, "/works", {"filter": f"authorships.author.id:{ids},from_publication_date:{FROM_YEAR}-01-01",
                                        "per-page": 200, "cursor": cursor, "select": WORK_FIELDS})
                recs += res.get("results", []); cursor = res.get("meta", {}).get("next_cursor")
                if not res.get("results"): break
            write_batch("works", run, n, recs); n += 1; done.update(batch)
            json.dump({"done": sorted(done)}, open(ck_path, "w")); print("works batch", n, len(recs), flush=True)
    except BudgetExhausted as e:
        print(f"OpenAlex budget exhausted; resets in {e}s. Re-run to resume.", flush=True)
    json.dump({"done": sorted(done)}, open(ck_path, "w"))


def linked_authors(min_votes=2):
    import glob
    out = []
    for f in glob.glob(os.path.join(LAND, "resolve", "*", "*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if not r["votes"]: continue
            best_id, best = max(r["votes"].items(), key=lambda kv: kv[1]["n"])
            if best["n"] >= min(min_votes, r["n_works"]): out.append(best_id)
    return sorted(set(out))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "resolve"
    if step == "resolve":
        step_resolve(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif step == "works":
        step_works(linked_authors())
