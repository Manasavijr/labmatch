"""Retrieval + ranking experiment with held-out author-paper links.

Protocol
--------
* For every lab with >= MIN_PAPERS grant-linked papers, hold out its most recent paper.
* The held-out paper is removed from *every* lab's profile (co-PIs share papers), and every lab
  linked to that paper counts as relevant.
* Query = held-out paper title. Labs are split 80/20 by id: LambdaRank trains on queries from
  train labs and is evaluated on queries from test labs, never seeing test labs' held-out papers.
* Systems compared on the test queries: BM25, SPECTER2 dense, hybrid (reciprocal rank fusion),
  and LightGBM LambdaRank re-ranking the hybrid top-100.

Writes data/build/eval.json and logs params/metrics to MLflow (./mlruns).
"""
import json, os, time
import duckdb, numpy as np, pandas as pd, scipy.sparse as sp
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = lambda *p: os.path.join(os.environ.get("LABMATCH_BUILD", os.path.join(ROOT, "data", "build")), *p)
MIN_PAPERS, K_CAND, SEED = 4, 100, 7
TODAY = pd.Timestamp("2026-11-12")


# ---------------------------------------------------------------- data
def load():
    con = duckdb.connect(os.path.join(ROOT, "data/lake/labmatch.duckdb"), read_only=True)
    prof = con.sql("select * from lab_profile order by inv_id").df()
    docs = con.sql("select inv_id, doc_type, doc_id, text, year from lab_document").df()
    return prof, docs


def holdout(docs):
    p = docs[docs.doc_type == "paper"]
    cnt = p.groupby("inv_id").size()
    elig = cnt[cnt >= MIN_PAPERS].index
    last = (p[p.inv_id.isin(elig)].sort_values(["year", "doc_id"], ascending=[False, False])
            .groupby("inv_id").head(1))
    held = last.drop_duplicates("doc_id")[["doc_id", "text", "year", "inv_id"]].rename(columns={"inv_id": "source_lab"})
    rel = p[p.doc_id.isin(held.doc_id)].groupby("doc_id").inv_id.apply(set)
    held["relevant"] = held.doc_id.map(rel)
    return held.reset_index(drop=True)


# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.4,
                                  max_features=600_000, dtype=np.float32)
        tf = self.cv.fit_transform(texts).tocsr()
        dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((tf.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom * idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms x labs

    def score(self, queries):
        q = self.cv.transform(queries); q.data[:] = 1
        return (q @ self.W).toarray()  # queries x labs


# ---------------------------------------------------------------- metrics
def metrics(ranked, relevant, ks=(10, 50, 100)):
    out = {f"recall@{k}": [] for k in ks}; out["ndcg@10"] = []; out["mrr"] = []
    for r, rel in zip(ranked, relevant):
        hits = np.array([x in rel for x in r[:max(ks)]])
        for k in ks:
            out[f"recall@{k}"].append(hits[:k].sum() / min(len(rel), k))
        g = hits[:10] / np.log2(np.arange(2, 12))
        ideal = (1 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
        out["ndcg@10"].append(g.sum() / ideal)
        pos = np.flatnonzero(hits)
        out["mrr"].append(1 / (pos[0] + 1) if len(pos) else 0.0)
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def topk(scores, k):
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


# ---------------------------------------------------------------- features
def lab_features(prof):
    f = pd.DataFrame(index=prof.index)
    f["log_active_amount"] = np.log1p(prof.active_amount.fillna(0))
    f["months_left"] = prof.months_funding_left.fillna(-1).clip(-1, 120)
    f["n_active"] = prof.n_active
    f["n_awards"] = prof.n_awards
    f["n_ml_awards"] = prof.n_ml_awards
    f["log_pubs"] = np.log1p(prof.n_pubs)
    f["log_pubs_recent"] = np.log1p(prof.n_pubs_recent)
    f["years_active"] = (TODAY - pd.to_datetime(prof.first_start)).dt.days.fillna(0) / 365
    f["years_since_last_award"] = (TODAY - pd.to_datetime(prof.last_start)).dt.days.fillna(9999) / 365
    f["is_nih"] = prof.agencies.fillna("").str.contains("NIH").astype(int)
    f["is_nsf"] = prof.agencies.fillna("").str.contains("NSF").astype(int)
    return f.values.astype(np.float32), list(f.columns)


def main():
    t0 = time.time()
    prof, docs = load()
    lab_index = {v: i for i, v in enumerate(prof.inv_id)}
    held = holdout(docs)
    print(len(prof), "labs,", len(held), "held-out papers", flush=True)

    # ---- profiles without held-out papers
    keep = docs[~((docs.doc_type == "paper") & docs.doc_id.isin(held.doc_id))]
    text = keep.groupby("inv_id").text.apply(lambda s: " \n".join(s.str[:3000]))
    lab_text = prof.inv_id.map(text).fillna("").values
    bm = BM25(lab_text)

    # ---- dense: lab centroid of doc embeddings, held-out excluded
    meta = pd.read_parquet(B("doc_meta.parquet")); E = np.load(B("doc_emb.npy")).astype(np.float32)
    m = meta[~((meta.doc_type == "paper") & meta.doc_id.isin(held.doc_id))]
    m = m[m.inv_id.isin(lab_index)]
    rows = m.inv_id.map(lab_index).values
    C = np.zeros((len(prof), E.shape[1]), np.float32)
    np.add.at(C, rows, E[m.row.values])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    qrow = meta[meta.doc_type == "paper"].drop_duplicates("doc_id").set_index("doc_id").row
    held = held[held.doc_id.isin(qrow.index)].reset_index(drop=True)
    Q = E[qrow.loc[held.doc_id].values]
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9

    # ---- split by source lab
    rng = np.random.default_rng(SEED)
    labs = held.source_lab.unique(); rng.shuffle(labs)
    test_labs = set(labs[: len(labs) // 5])
    held["split"] = np.where(held.source_lab.isin(test_labs), "test", "train")
    rel_idx = [set(lab_index[x] for x in r if x in lab_index) for r in held.relevant]

    # ---- score in chunks
    Fl, fnames = lab_features(prof)
    results = {"bm25": [], "dense": [], "hybrid": []}
    X, y, group, qsplit, cand_all = [], [], [], [], []
    for s in range(0, len(held), 1000):
        sl = slice(s, s + 1000)
        sb = bm.score(held.text.values[sl])
        sd = Q[sl] @ C.T
        rb, rd = topk(sb, K_CAND * 3), topk(sd, K_CAND * 3)
        for i in range(sb.shape[0]):
            qi = s + i
            results["bm25"].append(rb[i]); results["dense"].append(rd[i])
            rrf = {}
            for r, lab in enumerate(rb[i]): rrf[lab] = rrf.get(lab, 0) + 1 / (60 + r)
            for r, lab in enumerate(rd[i]): rrf[lab] = rrf.get(lab, 0) + 1 / (60 + r)
            cand = np.array(sorted(rrf, key=rrf.get, reverse=True)[:K_CAND])
            results["hybrid"].append(cand)
            pos_b = {lab: r for r, lab in enumerate(rb[i])}; pos_d = {lab: r for r, lab in enumerate(rd[i])}
            qf = np.stack([sb[i, cand], sd[i, cand], [rrf[c] for c in cand],
                           [pos_b.get(c, 999) for c in cand], [pos_d.get(c, 999) for c in cand],
                           np.full(len(cand), sb[i, cand].max()), np.full(len(cand), sd[i, cand].max())], 1)
            X.append(np.hstack([qf, Fl[cand]])); y.append(np.array([c in rel_idx[qi] for c in cand], int))
            group.append(len(cand)); qsplit.append(held.split[qi]); cand_all.append(cand)
        print("scored", min(s + 1000, len(held)), "/", len(held), f"{time.time() - t0:.0f}s", flush=True)
    names = ["bm25", "dense_cos", "rrf", "bm25_rank", "dense_rank", "bm25_qmax", "dense_qmax"] + fnames

    tr = [i for i, s in enumerate(qsplit) if s == "train"]; te = [i for i, s in enumerate(qsplit) if s == "test"]
    cat = lambda idx, arr: np.vstack([arr[i] for i in idx]) if arr is X else np.concatenate([arr[i] for i in idx])
    # LightGBM needs at least one positive per query group to learn from it
    tr_fit = [i for i in tr if y[i].any()]
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    va = tr_fit[: len(tr_fit) // 10]; fit = tr_fit[len(tr_fit) // 10:]
    dtr = lgb.Dataset(cat(fit, X), cat(fit, y), group=[group[i] for i in fit], feature_name=names)
    dva = lgb.Dataset(cat(va, X), cat(va, y), group=[group[i] for i in va], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])

    # Platt calibration on the validation queries: model score -> P(lab is relevant)
    from sklearn.linear_model import LogisticRegression
    vs = np.concatenate([model.predict(X[i]) for i in va]); vy = np.concatenate([y[i] for i in va])
    cal = LogisticRegression().fit(vs[:, None], vy)
    json.dump({"a": float(cal.coef_[0, 0]), "b": float(cal.intercept_[0])}, open(B("calib.json"), "w"))

    reranked = []
    for i in range(len(held)):
        if qsplit[i] != "test": reranked.append(None); continue
        sc = model.predict(X[i]); reranked.append(cand_all[i][np.argsort(-sc)])
    rel_te = [rel_idx[i] for i in te]
    report = {name: metrics([results[name][i] for i in te], rel_te) for name in ["bm25", "dense", "hybrid"]}
    report["lambdarank"] = metrics([reranked[i] for i in te], rel_te)
    imp = dict(sorted(zip(names, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
    out = dict(protocol=dict(min_papers=MIN_PAPERS, candidates=K_CAND, seed=SEED, labs=len(prof),
                             queries_total=len(held), queries_train=len(tr), queries_test=len(te),
                             best_iteration=model.best_iteration, query_text="held-out paper title"),
               metrics=report, feature_gain=imp)
    json.dump(out, open(B("eval.json"), "w"), indent=2)
    model.save_model(B("lambdarank.txt"))
    try:
        import mlflow
        mlflow.set_tracking_uri("file://" + os.path.join(ROOT, "mlruns"))
        mlflow.set_experiment("labmatch-ranker")
        with mlflow.start_run(run_name="heldout-paper-eval"):
            mlflow.log_params({**params, **{k: v for k, v in out["protocol"].items() if not isinstance(v, str)}})
            for sysname, ms in report.items():
                for k, v in ms.items(): mlflow.log_metric(f"{sysname}.{k.replace('@', '_at_')}", v)
            mlflow.log_artifact(B("eval.json")); mlflow.log_artifact(B("lambdarank.txt"))
    except Exception as e:
        print("mlflow logging skipped:", e)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
