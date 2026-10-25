#!/usr/bin/env python3
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

STOP = set("""a an and are as at be but by for from has have i if in into is it its me my of on or
so that the their them then there these they this to was we were what when where which who will with
would you your want like looking find place places near nice good great""".split())

E5_INSTRUCT = "Instruct: Given a search query, retrieve business reviews that match the query\nQuery: "


def read_jsonl(path, **kw):
    return pd.read_json(path, lines=True, dtype=False, convert_dates=False, **kw)


def tokenize(s):
    return [t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t not in STOP and len(t) > 1]


# ---------- data ----------
def select_pois(business_path, cats, city, state, min_reviews):
    df = read_jsonl(business_path)
    df["categories"] = df["categories"].fillna("")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    wanted = {c.strip().lower() for c in cats if c.strip()}
    df = df[df["categories"].map(lambda s: any(c.strip().lower() in wanted for c in s.split(",")))]
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    df = df[df["review_count"] >= min_reviews].reset_index(drop=True)
    return df[["business_id", "name", "city", "state", "categories", "review_count",
               "latitude", "longitude"]]


def load_reviews(reviews_path, ids, cache):
    if cache and os.path.exists(cache):
        print(f"Pakai cache review: {cache}")
        return read_jsonl(cache)
    parts = []
    for ch in read_jsonl(reviews_path, chunksize=200_000):
        ch = ch[ch["business_id"].isin(ids)][["business_id", "review_id", "text"]]
        if len(ch):
            parts.append(ch)
    rv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["business_id", "review_id", "text"])
    if cache:
        rv.to_json(cache, orient="records", lines=True, force_ascii=False)
    return rv


def cap_reviews(rv, n, seed=0):
    if n <= 0:
        return rv
    rv = rv.sample(frac=1.0, random_state=seed)
    return rv.groupby("business_id", sort=False).head(n).reset_index(drop=True)


# ---------- BM25 ----------
def bm25_scores(pois, rv, queries):
    from rank_bm25 import BM25Okapi
    by_poi = rv.groupby("business_id")["text"].apply(lambda s: " ".join(s)).to_dict()
    docs = []
    for _, r in pois.iterrows():
        txt = f'{r["name"]} {r["categories"]} {by_poi.get(r["business_id"], "")}'
        docs.append(tokenize(txt))
    bm = BM25Okapi(docs)
    return np.vstack([bm.get_scores(tokenize(q)) for q in queries])  # [n_query, n_poi]


# ---------- dense ----------
def load_encoder(name):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.half()  # T4: fp16, bukan bf16
    model.max_seq_length = 256

    def enc(texts, is_query=False):
        if is_query and "e5" in name.lower() and "instruct" in name.lower():
            texts = [E5_INSTRUCT + t for t in texts]
        elif "e5" in name.lower() and "instruct" not in name.lower():
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        return model.encode(texts, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=not is_query)
    return enc


def dense_scores(enc, pois, rv, queries, pool="max", emb_cache=None):
    pid = {b: i for i, b in enumerate(pois["business_id"])}
    rv = rv[rv["business_id"].isin(pid)].copy()
    rv["pi"] = rv["business_id"].map(pid)
    rv = rv.sort_values("pi").reset_index(drop=True)

    if emb_cache and os.path.exists(emb_cache):
        R = np.load(emb_cache)
        print(f"Pakai cache embedding: {emb_cache}")
    else:
        R = np.asarray(enc(rv["text"].tolist(), is_query=False), dtype=np.float32)
        if emb_cache:
            np.save(emb_cache, R)
    Q = np.asarray(enc(list(queries), is_query=True), dtype=np.float32)
    sims = Q @ R.T                                   # [n_query, n_review]
    pi = rv["pi"].to_numpy()
    n_poi = len(pois)
    out = np.full((len(queries), n_poi), -1.0, dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    owners = pi[starts]
    if pool == "max":
        out[:, owners] = np.maximum.reduceat(sims, starts, axis=1)
    else:  # top3 mean
        ends = np.r_[starts[1:], len(pi)]
        for k, (s, e) in enumerate(zip(starts, ends)):
            seg = np.sort(sims[:, s:e], axis=1)[:, ::-1][:, :3]
            out[:, owners[k]] = seg.mean(axis=1)
    return out


# ---------- fusi & output ----------
def rrf(score_mats, k=60):
    fused = np.zeros_like(score_mats[0], dtype=np.float64)
    for S in score_mats:
        ranks = (-S).argsort(axis=1).argsort(axis=1) + 1
        fused += 1.0 / (k + ranks)
    return fused


def topk_rows(qids, pois, S, method, k):
    rows = []
    for qi, qid in enumerate(qids):
        idx = np.argsort(-S[qi])[:k]
        for r, j in enumerate(idx, 1):
            rows.append({"query_id": qid, "method": method, "rank": r,
                         "business_id": pois.iloc[j]["business_id"],
                         "name": pois.iloc[j]["name"], "score": float(S[qi, j])})
    return rows


def build_pool(runs, qdf, pois, pool_k):
    p = runs[runs["rank"] <= pool_k]
    g = p.groupby(["query_id", "business_id"]).agg(
        methods=("method", lambda s: "|".join(sorted(set(s)))),
        best_rank=("rank", "min")).reset_index()
    g = g.merge(qdf[["id", "text"]].rename(columns={"id": "query_id", "text": "query"}), on="query_id")
    g = g.merge(pois[["business_id", "name", "city", "categories"]], on="business_id")
    for c in ("label_a1", "label_a2", "label_a3"):
        g[c] = ""
    return g.sort_values(["query_id", "best_rank"])[
        ["query_id", "query", "business_id", "name", "city", "categories",
         "methods", "best_rank", "label_a1", "label_a2", "label_a3"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business", required=True)
    ap.add_argument("--reviews", help="review.json (atau file cache dari --review-cache)")
    ap.add_argument("--cats", required=True)
    ap.add_argument("--city")
    ap.add_argument("--state")
    ap.add_argument("--min-reviews", type=int, default=10)
    ap.add_argument("--max-reviews-per-poi", type=int, default=100)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--dense", default="intfloat/multilingual-e5-large-instruct",
                    help="nama model sentence-transformers, atau 'none'")
    ap.add_argument("--pool", choices=["max", "top3"], default="max")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--pool-k", type=int, default=10)
    ap.add_argument("--review-cache")
    ap.add_argument("--outdir", default="out_text")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cats = [c.strip() for c in args.cats.split(",")]
    pois = select_pois(args.business, cats, args.city, args.state, args.min_reviews)
    if pois.empty:
        sys.exit("Tidak ada POI yang cocok. Cek --cats/--city/--state.")
    print(f"POI terpilih: {len(pois)}")

    rv = load_reviews(args.reviews, set(pois["business_id"]), args.review_cache)
    rv = cap_reviews(rv, args.max_reviews_per_poi)
    print(f"Review dipakai: {len(rv)} (maks {args.max_reviews_per_poi} per POI)")

    qdf = pd.read_csv(args.queries)
    qids, qtexts = qdf["id"].tolist(), qdf["text"].tolist()

    mats = {"bm25": bm25_scores(pois, rv, qtexts)}
    if args.dense.lower() != "none":
        enc = load_encoder(args.dense)
        emb_cache = os.path.join(args.outdir, "review_emb.npy")
        mats["dense"] = dense_scores(enc, pois, rv, qtexts, pool=args.pool, emb_cache=emb_cache)
        mats["rrf"] = rrf([mats["bm25"], mats["dense"]])

    rows = []
    for name, S in mats.items():
        rows += topk_rows(qids, pois, S, name, args.topk)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.outdir, "runs_text.csv"), index=False)
    pool = build_pool(runs, qdf, pois, args.pool_k)
    pool.to_csv(os.path.join(args.outdir, "pool_for_annotation.csv"), index=False)

    print(f"\nTersimpan di {args.outdir}/")
    print(f"  runs_text.csv            : top-{args.topk} per query per metode")
    print(f"  pool_for_annotation.csv  : {len(pool)} pasangan unik (top-{args.pool_k} gabungan)")


if __name__ == "__main__":
    main()