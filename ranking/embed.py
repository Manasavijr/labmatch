"""Embed lab documents (grant title+abstract, grant-linked paper titles) with SPECTER2.

Reads the dbt mart `lab_document`; writes data/build/doc_meta.parquet + doc_emb.npy (float16, L2-normalised).
Papers are capped to the 15 most recent per lab to keep CPU encoding to about an hour.
"""
import os, sys, time
import duckdb, numpy as np, torch
from sentence_transformers import SentenceTransformer, models

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.environ.get("SPECTER2_PATH", os.path.join(ROOT, "data/models/specter2_base"))


def load_docs():
    con = duckdb.connect(os.path.join(ROOT, "data/lake/labmatch.duckdb"), read_only=True)
    return con.sql("""
        with g as (
            -- AI/ML grants keep title + abstract; other grants of the same PI contribute their title only
            select d.inv_id, d.doc_type, d.doc_id,
                   case when a.is_ml then d.text else a.title end as text, d.year
            from lab_document d
            join (select distinct award_id, is_ml, title from int_lab_award) a on cast(a.award_id as varchar) = d.doc_id
            where d.doc_type = 'grant'
        ), p as (
            select inv_id, doc_type, doc_id, text, year,
                   row_number() over (partition by inv_id order by year desc nulls last, doc_id) as rn
            from lab_document where doc_type = 'paper'
        )
        select * from g union all select inv_id, doc_type, doc_id, text, year from p where rn <= 15
    """).df()


def encoder(max_len):
    torch.set_num_threads(os.cpu_count())
    w = models.Transformer(MODEL, max_seq_length=max_len)
    return SentenceTransformer(modules=[w, models.Pooling(w.get_word_embedding_dimension(), pooling_mode="cls")], device="cpu")


def main():
    docs = load_docs()
    # identical texts (e.g. a paper credited to several labs) are encoded once
    uniq = docs[["doc_type", "doc_id", "text"]].drop_duplicates(["doc_type", "doc_id"]).reset_index(drop=True)
    print(len(docs), "doc rows,", len(uniq), "unique", flush=True)
    embs = np.zeros((len(uniq), 768), dtype=np.float16)
    short = uniq.text.str.len().values < 300
    plan = [("paper", 48, 128, uniq.doc_type.values == "paper"),
            ("grant-title", 48, 128, (uniq.doc_type.values == "grant") & short),
            ("grant-abstract", 128, 64, (uniq.doc_type.values == "grant") & ~short)]
    for kind, max_len, bs, maskk in plan:
        idx = np.where(maskk)[0]
        cache = os.path.join(ROOT, f"data/build/_emb_{kind}.npy")
        if os.path.exists(cache) and np.load(cache).shape[0] == len(idx):
            embs[idx] = np.load(cache); print(kind, "cached", flush=True); continue
        m = encoder(max_len)
        t = time.time()
        for s in range(0, len(idx), 4096):
            part = idx[s:s + 4096]
            e = m.encode(uniq.text.values[part].tolist(), batch_size=bs, normalize_embeddings=True)
            embs[part] = e.astype(np.float16)
            print(kind, s + len(part), "/", len(idx), f"{(s + len(part)) / (time.time() - t):.0f}/s", flush=True)
        np.save(cache, embs[idx])
    uniq["row"] = np.arange(len(uniq))
    docs = docs.merge(uniq[["doc_type", "doc_id", "row"]], on=["doc_type", "doc_id"])
    os.makedirs(os.path.join(ROOT, "data/build"), exist_ok=True)
    docs.drop(columns=["text"]).to_parquet(os.path.join(ROOT, "data/build/doc_meta.parquet"))
    uniq[["doc_type", "doc_id", "text"]].to_parquet(os.path.join(ROOT, "data/build/doc_text.parquet"))
    np.save(os.path.join(ROOT, "data/build/doc_emb.npy"), embs)


if __name__ == "__main__":
    main()
