"""v0 lab profiles: one row per PI who holds an AI/ML grant at a US institution.

Later replaced by the dbt `lab_profile` model; same columns.
"""
import re
import pandas as pd
from ranking import load

TODAY = pd.Timestamp("2026-11-12")
ML = re.compile(r"machine learning|deep learning|neural network|artificial intelligence|reinforcement learning|"
                r"large language model|natural language processing|computer vision|transformer model|"
                r"foundation model|generative model", re.I)


def titlecase_name(n):
    if not isinstance(n, str):
        return n
    if "," in n:
        last, first = n.split(",", 1)
        n = f"{first.strip()} {last.strip()}"
    n = " ".join(w.capitalize() if w.isupper() or w.islower() else w for w in n.split())
    n = re.sub(r"-([a-z])", lambda m: "-" + m.group(1).upper(), n)
    return re.sub(r"\((\w)", lambda m: "(" + m.group(1).upper(), n)


def main():
    a = load.awards()
    inst = load.institutions()
    us = set(inst.loc[inst.country.fillna("").str.upper().isin(["UNITED STATES", "USA", "US"]), "inst_id"])
    a = a[a.inst_id.isin(us)]
    a["is_ml"] = (a.title.fillna("") + " " + a.abstract.fillna("")).str.contains(ML)
    ai = load.award_investigator()
    inv = load.investigators()
    ml_pis = set(ai.merge(a[a.is_ml][["award_id"]], on="award_id").inv_id)
    ai = ai[ai.inv_id.isin(ml_pis)].merge(a, on="award_id")
    ai["active"] = ai.end >= TODAY

    pubs = load.publications()
    ap = load.award_publication().merge(pubs, on="pub_id")
    pi_pubs = ai[["inv_id", "award_id"]].merge(ap, on="award_id").drop_duplicates(["inv_id", "pub_id"])

    rows = []
    pubs_by = {k: g for k, g in pi_pubs.groupby("inv_id")}
    for inv_id, g in ai.groupby("inv_id"):
        g = g.sort_values("start", ascending=False)
        act = g[g.active]
        p = pubs_by.get(inv_id)
        ptitles = [] if p is None else p.sort_values("pub_year", ascending=False).title.dropna().tolist()
        rows.append(dict(
            inv_id=inv_id,
            n_awards=len(g), n_ml_awards=int(g.is_ml.sum()), n_active=len(act),
            active_amount=float(act.amount.fillna(0).sum()),
            active_end_max=act.end.max(), first_start=g.start.min(), last_start=g.start.max(),
            agencies="/".join(sorted(g.agency.dropna().unique())),
            n_pubs=0 if p is None else len(p),
            n_pubs_recent=0 if p is None else int((p.pub_year >= 2023).sum()),
            award_ids=g.award_id.tolist(),
            text=" \n".join((g.title.fillna("") + ". " + g.abstract.fillna("").str[:2500]).tolist() + ptitles[:60]),
            pub_titles=ptitles[:8],
        ))
    prof = pd.DataFrame(rows).merge(inv[["inv_id", "full_name", "inst_id", "nih_profile_id", "nsf_id"]], on="inv_id")
    prof = prof.merge(inst.rename(columns={"name": "inst_name"}), on="inst_id", how="left")
    prof["name"] = prof.full_name.map(titlecase_name)
    prof.to_parquet("data/build/lab_profile_v0.parquet")
    keep = a[a.award_id.isin(set(ai.award_id))][["award_id", "agency", "src", "title", "abstract", "program", "start", "end", "amount", "is_ml"]]
    keep.to_parquet("data/build/award_ml_v0.parquet")
    print(len(prof), "labs;", int((prof.n_active > 0).sum()), "with active funding;", len(keep), "awards")


if __name__ == "__main__":
    main()
