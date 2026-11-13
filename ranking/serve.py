"""Production ranker: BM25 + SPECTER2 hybrid retrieval, LightGBM LambdaRank re-ranking, explanation per lab."""
import os
import duckdb, numpy as np, pandas as pd, lightgbm as lgb, torch
from sentence_transformers import SentenceTransformer, models
from ranking.experiment import BM25, lab_features, topk, ROOT, B, K_CAND
from ranking.retrieve_v0 import tc, TODAY
from ranking.build_profiles import titlecase_name


class Ranker:
    def __init__(self):
        con = duckdb.connect(os.path.join(ROOT, "data/lake/labmatch.duckdb"), read_only=True)
        self.p = con.sql("select * from lab_profile order by inv_id").df()
        docs = con.sql("select inv_id, doc_type, doc_id, text, year from lab_document").df()
        self.aw = con.sql("select award_id, agency, source_award_no as src, title, abstract, start_date as start, end_date as \"end\", total_amount as amount from int_lab_award").df().drop_duplicates("award_id").set_index("award_id")
        self.lab_awards = con.sql("select inv_id, list(award_id) a from int_lab_award group by 1").df().set_index("inv_id").a
        self.lab_pubs = con.sql("select inv_id, list(title order by pub_year desc) t from int_lab_publication group by 1").df().set_index("inv_id").t
        idx = {v: i for i, v in enumerate(self.p.inv_id)}
        text = docs.groupby("inv_id").text.apply(lambda s: " \n".join(s.str[:3000]))
        self.bm = BM25(self.p.inv_id.map(text).fillna("").values)
        meta = pd.read_parquet(B("doc_meta.parquet")); E = np.load(B("doc_emb.npy")).astype(np.float32)
        meta = meta[meta.inv_id.isin(idx)]
        self.C = np.zeros((len(self.p), E.shape[1]), np.float32)
        np.add.at(self.C, meta.inv_id.map(idx).values, E[meta.row.values])
        self.C /= np.linalg.norm(self.C, axis=1, keepdims=True) + 1e-9
        self.F, _ = lab_features(self.p)
        self.model = lgb.Booster(model_file=B("lambdarank.txt"))
        w = models.Transformer(os.environ.get("SPECTER2_PATH", os.path.join(ROOT, "data/models/specter2_base")), max_seq_length=256)
        self.enc = SentenceTransformer(modules=[w, models.Pooling(w.get_word_embedding_dimension(), pooling_mode="cls")], device="cpu")
        import json
        self.cal = json.load(open(B("calib.json")))

    def search(self, q, k=10, active_only=True):
        sb = self.bm.score([q])[0]
        qv = self.enc.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        sd = self.C @ qv
        if active_only:
            mask = (self.p.n_active.values > 0) & (self.p.active_amount.values >= 1e5)
            sb = np.where(mask, sb, -1e9); sd = np.where(mask, sd, -1e9)
        rb, rd = topk(sb[None], K_CAND * 3)[0], topk(sd[None], K_CAND * 3)[0]
        rrf = {}
        for r, lab in enumerate(rb): rrf[lab] = rrf.get(lab, 0) + 1 / (60 + r)
        for r, lab in enumerate(rd): rrf[lab] = rrf.get(lab, 0) + 1 / (60 + r)
        cand = np.array(sorted(rrf, key=rrf.get, reverse=True)[:K_CAND])
        pb = {lab: r for r, lab in enumerate(rb)}; pd_ = {lab: r for r, lab in enumerate(rd)}
        X = np.hstack([np.stack([sb[cand], sd[cand], [rrf[c] for c in cand], [pb.get(c, 999) for c in cand], [pd_.get(c, 999) for c in cand],
                                 np.full(len(cand), sb[cand].max()), np.full(len(cand), sd[cand].max())], 1), self.F[cand]])
        s = self.model.predict(X)
        order = np.argsort(-s)[:k]
        prob = 1 / (1 + np.exp(-(self.cal["a"] * s + self.cal["b"])))
        qterms = list(dict.fromkeys(self.bm.cv.build_analyzer()(q)))
        out = []
        for rank, j in enumerate(order, 1):
            i = cand[j]; r = self.p.iloc[i]
            aws = self.aw.loc[[a for a in self.lab_awards.get(r.inv_id, []) if a in self.aw.index]]
            act = aws[pd.to_datetime(aws["end"]) >= TODAY].sort_values("end", ascending=False)
            aws_r = aws[~aws.title.fillna('').str.lower().str.contains('i-corps|conference|workshop|travel|symposium')]
            best = aws_r.assign(o=(aws_r.title.fillna("") + " " + aws_r.abstract.fillna("")).str.lower().map(lambda t: sum(w in t for w in qterms))).sort_values("o", ascending=False)
            col = self.bm.W[:, i]
            hits = sorted([(col[self.bm.cv.vocabulary_[t], 0], t) for t in qterms if t in self.bm.cv.vocabulary_ and col[self.bm.cv.vocabulary_[t], 0] > 0], reverse=True)
            shared = []
            for _, term in sorted(hits, key=lambda h: (-(" " in h[1]), -h[0])):
                if not any(term in s_ or s_ in term for s_ in shared): shared.append(term)
            shared = shared[:3]
            out.append(dict(rank=rank, inv_id=int(r.inv_id), name=titlecase_name(r.full_name), institution=tc(r.inst_name),
                            city=(r.city or "").title(), state=r.state, agencies=r.agencies,
                            score=round(float(prob[j]), 4),
                            signals=dict(fit=round(float(np.clip((sd[i] - 0.6) / 0.35, 0, 1)), 3), funding=round(float(min(1, np.log1p(r.active_amount) / np.log1p(5e7))), 3),
                                         recency=round(min(r.n_pubs_recent, 30) / 30, 3)),
                            active_amount=float(r.active_amount), n_active=int(r.n_active), n_pubs_recent=int(r.n_pubs_recent),
                            funded_until=None if pd.isna(r.active_end_max) else str(r.active_end_max)[:10],
                            why=dict(terms=shared, grant=best.title.iloc[0] if len(best) else None, grant_agency=best.agency.iloc[0] if len(best) else None),
                            grants=[dict(title=g.title, agency=g.agency, src=g.src, end=str(g.end)[:10], amount=g.amount) for g in act.head(4).itertuples()],
                            papers=list(self.lab_pubs.get(r.inv_id, []))[:4]))
        return out
