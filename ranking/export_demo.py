"""Export a compact static index for the shareable LabMatch page (top labs + example rankings)."""
import json, re
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from ranking.retrieve_v0 import Index, tc, TODAY
from ranking import load

EX = ["Graph neural networks for protein structure and drug discovery",
      "Reinforcement learning for robot manipulation",
      "LLMs for clinical notes and medical question answering",
      "Fairness and privacy in machine learning",
      "Computer vision for medical imaging and radiology"]
N = 4000

import duckdb, os
from ranking.serve import Ranker
idx = Index(); p = idx.p
R = Ranker()
pre = {q: R.search(q, k=10) for q in EX}
dbtp = R.p.set_index("inv_id")
p = p[p.inv_id.isin(dbtp.index)].copy()
for col in ["n_active", "active_amount", "active_end_max"]:
    p[col] = p.inv_id.map(dbtp[col])
p["active_end_max"] = pd.to_datetime(p.active_end_max)
need = {r["inv_id"] for v in pre.values() for r in v}
act = p[(p.n_active > 0) & (p.active_amount >= 1e5)].copy()
act["prio"] = act.n_ml_awards * 2 + np.log1p(act.active_amount) + np.log1p(act.n_pubs_recent)
sel_ids = list(dict.fromkeys(list(act.sort_values("prio", ascending=False).inv_id.head(N)) + list(need)))
sel_ids = [v for v in sel_ids if v in set(p.inv_id)]
xpos = {v: i for i, v in enumerate(idx.p.inv_id)}
X = idx.X[np.array([xpos[v] for v in sel_ids])].tocsr()
p = p.set_index("inv_id", drop=False)

# 2-D research landscape: TF-IDF -> SVD(64) -> t-SNE
Z = TruncatedSVD(64, random_state=0).fit_transform(X)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
xy = TSNE(2, init="pca", random_state=0, perplexity=35, metric="cosine").fit_transform(Z)
xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

fund = load.read("award_funding", ["award_id", "fiscal_year", "amount"])
ap = load.award_publication().merge(load.publications()[["pub_id", "title", "pub_year", "journal"]], on="pub_id")
ai = load.award_investigator()
inv = load.investigators()[["inv_id", "full_name"]]
from ranking.build_profiles import titlecase_name

vocab, labs = {}, []
for row, v in enumerate(sel_ids):
    r = p.loc[v]
    xr = X[row]
    top = np.argsort(-xr.data)[:28]
    terms = []
    for t in top:
        term = idx.terms[xr.indices[t]]
        terms.append([vocab.setdefault(term, len(vocab)), round(float(xr.data[t]), 3)])
    aw = idx.aw.loc[[a for a in r.award_ids if a in idx.aw.index]].copy()
    aw["active"] = aw.end >= TODAY
    aw = aw[~aw.title.fillna("").str.lower().str.contains("i-corps|conference|workshop|travel|symposium|summer school")]
    aw = aw.sort_values(["active", "end"], ascending=False)
    fy = fund[fund.award_id.isin(aw.index)].groupby("fiscal_year").amount.sum()
    pubs = ap[ap.award_id.isin(aw.index)].drop_duplicates("pub_id").sort_values("pub_year", ascending=False)
    co = ai[ai.award_id.isin(aw.index) & (ai.inv_id != r.inv_id)].inv_id.value_counts().head(5)
    co_names = [titlecase_name(n) for n in inv.set_index("inv_id").loc[co.index].full_name] if len(co) else []
    labs.append(dict(
        id=int(r.inv_id), n=titlecase_name(r.full_name), i=tc(r.inst_name), c=(r.city or "").title(), s=r.state, a=r.agencies,
        m=float(r.active_amount), u=None if pd.isna(r.active_end_max) else r.active_end_max.strftime("%Y-%m-%d"),
        pr=int(r.n_pubs_recent), np=int(r.n_pubs), na=int(r.n_awards), nact=int(r.n_active),
        since=None if pd.isna(r.first_start) else int(r.first_start.year),
        g=[[g.title, g.agency, g.src, None if pd.isna(g.amount) else float(g.amount),
            None if pd.isna(g.start) else g.start.strftime("%Y-%m-%d"), None if pd.isna(g.end) else g.end.strftime("%Y-%m-%d"),
            (g.program or "")] for g in aw.head(6).itertuples()],
        p=[[q.title, None if pd.isna(q.pub_year) else int(q.pub_year), q.journal] for q in pubs.head(6).itertuples()],
        fy=[[int(y), round(float(v))] for y, v in fy.items() if 2010 <= y <= 2027 and v > 0],
        co=co_names, xy=[round(float(xy[row, 0]), 4), round(float(xy[row, 1]), 4)], t=terms))

pre_ids = {q: [{"id": r["inv_id"], "score": r["score"], "sig": r["signals"], "terms": r["why"]["terms"],
                "grant": r["why"]["grant"], "ga": r["why"]["grant_agency"]} for r in v] for q, v in pre.items()}
# each award counted once (multi-PI awards appear once per PI in int_lab_award), active research AI/ML grants of funded labs only
ACTIVE_ML_DOLLARS = float(duckdb.connect(os.path.join(ROOT, "data/lake/labmatch.duckdb"), read_only=True).sql("""
    with f as (select inv_id from lab_profile where n_active > 0 and active_amount >= 1e5)
    select sum(total_amount) from (select distinct award_id, total_amount from int_lab_award join f using (inv_id) where is_active and is_research and is_ml)
""").fetchall()[0][0])
tl = json.load(open("data/raw/transform_log.json"))
stats = dict(labs=int(len(dbtp)), funded_labs=int(((dbtp.n_active > 0) & (dbtp.active_amount >= 1e5)).sum()), demo_labs=len(labs),
             grants_ml=int(len(idx.aw)), active_dollars=ACTIVE_ML_DOLLARS,
             raw_awards=366635, raw_investigators=219155, raw_pub_links=2355951, raw_pubs=969422,
             matched_cross_agency=next((s.get("matched") for s in tl["steps"] if s["step"] == "match_investigators"), None))
s = json.dumps(dict(stats=stats, vocab=list(vocab), labs=labs, pre=pre_ids), separators=(",", ":"), default=str)
s = re.sub(r'(?<=[,:\[])NaN(?=[,\]}])', 'null', s)
open("data/build/demo.json", "w").write(s)
print(stats, len(vocab), len(s))
