"""v0 retrieval: TF-IDF (BM25-style sublinear tf) over lab profiles. Replaced by SPECTER2 + BM25 hybrid."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TODAY = pd.Timestamp("2026-11-12")
SMALL = {"of", "at", "and", "the", "for", "in", "de"}


def tc(s):
    w = (s or "").title().split(" ")
    return " ".join(x.lower() if i and x.lower() in SMALL else x for i, x in enumerate(w))


class Index:
    def __init__(self, prof_path="data/build/lab_profile_v0.parquet", award_path="data/build/award_ml_v0.parquet"):
        self.p = pd.read_parquet(prof_path).reset_index(drop=True)
        self.aw = pd.read_parquet(award_path).set_index("award_id")
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=3, max_df=0.4,
                                   ngram_range=(1, 2), max_features=400_000, dtype=np.float32)
        self.X = self.vec.fit_transform(self.p.text.values)
        self.terms = np.array(self.vec.get_feature_names_out())

    def search(self, q, k=12, active_only=True):
        qv = self.vec.transform([q])
        s = (self.X @ qv.T).toarray().ravel()
        if active_only:
            s = np.where(self.p.n_active.values > 0, s, 0)
        cand = np.argsort(-s)[:300]
        fit = s[cand] / (s[cand].max() + 1e-9)
        fund = np.log1p(self.p.active_amount.values[cand]) / np.log1p(5e7)
        rec = np.minimum(self.p.n_pubs_recent.values[cand], 30) / 30
        score = 0.75 * fit + 0.15 * np.clip(fund, 0, 1) + 0.10 * rec
        order = np.argsort(-score)[:k]
        out = []
        qnz = set(qv.nonzero()[1])
        for rank, j in enumerate(order, 1):
            i = cand[j]
            r = self.p.iloc[i]
            row = self.X[i]
            shared = [(row[0, t], t) for t in qnz if row[0, t] > 0]
            shared = [self.terms[t] for _, t in sorted(shared, reverse=True)[:5]]
            out.append(self._card(r, rank, float(score[j]), float(fit[j]), float(np.clip(fund[j], 0, 1)), float(rec[j]), shared, qv))
        return out

    def _card(self, r, rank, score, fit, fund, rec, shared, qv):
        aw = self.aw.loc[[a for a in r.award_ids if a in self.aw.index]].copy()
        aw["active"] = aw.end >= TODAY
        tv = self.vec.transform((aw.title.fillna("") + ". " + aw.abstract.fillna("").str[:2500]).values)
        aw["sim"] = (tv @ qv.T).toarray().ravel()
        best = aw.sort_values(["sim"], ascending=False).iloc[0]
        act = aw[aw.active].sort_values("end", ascending=False)
        return dict(
            rank=rank, inv_id=int(r.inv_id), name=r["name"], institution=tc(r.inst_name),
            city=(r.city or "").title(), state=r.state, agencies=r.agencies,
            score=round(score, 4), signals=dict(fit=round(fit, 3), funding=round(fund, 3), recency=round(rec, 3)),
            active_amount=r.active_amount, n_active=int(r.n_active), n_pubs_recent=int(r.n_pubs_recent),
            funded_until=None if pd.isna(r.active_end_max) else r.active_end_max.strftime("%Y-%m-%d"),
            why=dict(terms=shared, grant=best.title, grant_agency=best.agency),
            grants=[dict(title=g.title, agency=g.agency, src=g.src, start=g.start.strftime("%Y-%m-%d") if pd.notna(g.start) else None,
                         end=g.end.strftime("%Y-%m-%d") if pd.notna(g.end) else None, amount=g.amount)
                    for g in act.head(4).itertuples()],
            papers=list(r.pub_titles[:4]),
        )
