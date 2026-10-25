#!/usr/bin/env python3
"""
Baseline teks untuk POI retrieval di Yelp: BM25, dense (embedding review), dan RRF hybrid.
Output: top-k per query per metode, plus "pool" gabungan untuk annotator.

Persiapan (Kaggle: GPU T4 hanya perlu untuk dense; BM25 jalan di CPU):
    pip install rank_bm25 sentence-transformers pandas

File query (CSV), kolom wajib id,text ; kolom type opsional:
    id,text,type
    P1,"I want to relax while camping near a river with shady forest",mixed
    ...

Contoh:
    python baseline_text.py \
        --business /kaggle/input/.../yelp_academic_dataset_business.json \
        --reviews  /kaggle/input/.../yelp_academic_dataset_review.json \
        --cats "Parks,Campgrounds,Beaches" --city Boise --state ID \
        --queries queries.csv --outdir out_text

    Multi-metro: --state IN,FL,LA (beberapa state sekaligus; satu state Yelp ~ satu metro).
    Tambahkan kolom `state` di queries.csv (mis. TN) supaya tiap query hanya mencari di metronya.
    Tanpa --cats = korpus satu kota penuh (semua kategori), lebih realistis untuk query bebas.
    --dense none           lewati dense (BM25 saja, tanpa GPU)
    --review-cache f.jsonl simpan subset review supaya run berikutnya tidak baca 5 GB lagi

Catatan desain:
- BM25 bekerja di level POI (nama + kategori + gabungan review).
- Dense meng-embed tiap review, lalu skor POI = max (atau top-3 mean) kemiripan antar review POI itu.
  Ini keputusan "pooling" yang bisa lu jadikan baris ablasi.
- Atribut Yelp (Ambience, NoiseLevel, dst.) SENGAJA tidak dipakai di sini, supaya bisa
  dipakai sebagai ground truth otomatis nanti.
"""
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
    if wanted:  # tanpa --cats: semua kategori (korpus satu kota penuh)
        df = df[df["categories"].map(lambda s: any(c.strip().lower() in wanted for c in s.split(",")))]
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if state:  # boleh beberapa state dipisah koma, mis. "IN,FL,LA"
        states = {x.strip().lower() for x in state.split(",") if x.strip()}
        df = df[df["state"].str.lower().isin(states)]
    df = df[df["review_count"] >= min_reviews].reset_index(drop=True)
    return df[["business_id", "name", "city", "state", "categories", "review_count",
               "latitude", "longitude"]]


def load_reviews(reviews_path, ids, cache):
    if cache and os.path.exists(cache):
        cached = read_jsonl(cache)
        cached = cached[cached["business_id"].isin(ids)]
        covered = cached["business_id"].nunique() / max(len(ids), 1)
        if covered >= 0.8:  # cache cukup mewakili POI yang diminta
            print(f"Pakai cache review: {cache} (mencakup {covered:.0%} POI)")
            return cached.reset_index(drop=True)
        print(f"Cache {cache} hanya mencakup {covered:.0%} POI yang diminta "
              f"(kemungkinan dibuat untuk scope lain). Baca ulang review.json dan timpa cache.")
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

    import hashlib
    fp = hashlib.md5("|".join(rv["review_id"].astype(str)).encode()).hexdigest()
    fp_path = (emb_cache + ".fp") if emb_cache else None
    R = None
    if emb_cache and os.path.exists(emb_cache) and os.path.exists(fp_path):
        if open(fp_path).read().strip() == fp:
            R = np.load(emb_cache)
            print(f"Pakai cache embedding: {emb_cache}")
        else:
            print("Cache embedding dibuat untuk set review yang berbeda -> hitung ulang.")
    if R is None:
        R = np.asarray(enc(rv["text"].tolist(), is_query=False), dtype=np.float32)
        if emb_cache:
            np.save(emb_cache, R)
            open(fp_path, "w").write(fp)
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


# ---------- filter metro/state per query ----------
def metro_mask(qdf, pois):
    """Boolean [n_query, n_poi]. Kolom 'state' di queries.csv (mis. TN) membatasi kandidat
    ke state itu; kosong atau 'ALL' = tanpa batas. Semua baseline dapat filter yang SAMA."""
    n_q, n_p = len(qdf), len(pois)
    if "state" not in qdf.columns:
        return np.ones((n_q, n_p), dtype=bool)
    ps = pois["state"].str.lower().to_numpy()
    mask = np.ones((n_q, n_p), dtype=bool)
    for i, v in enumerate(qdf["state"].fillna("").astype(str)):
        v = v.strip().lower()
        if v and v != "all":
            mask[i] = ps == v
    return mask


def apply_mask(S, mask, fill=-1e9):
    return np.where(mask, S, fill)


# ---------- fusi & output ----------
def rrf(score_mats, k=60):
    fused = np.zeros_like(score_mats[0], dtype=np.float64)
    for S in score_mats:
        ranks = (-S).argsort(axis=1).argsort(axis=1) + 1
        fused += 1.0 / (k + ranks)
    return fused


def topk_rows(qids, pois, S, method, k, mask=None):
    rows = []
    for qi, qid in enumerate(qids):
        order = np.argsort(-S[qi])
        if mask is not None:  # buang POI di luar metro query, jangan jadi pengisi top-k
            order = order[mask[qi][order]]
        idx = order[:k]
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
    g = g.merge(pois[["business_id", "name", "city", "state", "categories"]], on="business_id")
    for c in ("label_a1", "label_a2", "label_a3"):
        g[c] = ""
    return g.sort_values(["query_id", "best_rank"])[
        ["query_id", "query", "business_id", "name", "city", "state", "categories",
         "methods", "best_rank", "label_a1", "label_a2", "label_a3"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business", required=True)
    ap.add_argument("--reviews", help="review.json (atau file cache dari --review-cache)")
    ap.add_argument("--cats", default="",
                    help="filter kategori (pisah koma). Kosongkan = semua kategori di kota itu")
    ap.add_argument("--city", nargs="+", help="nama kota; boleh lebih dari satu kata, mis. --city New Orleans")
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
    args.city = " ".join(args.city) if args.city else None
    os.makedirs(args.outdir, exist_ok=True)

    cats = [c.strip() for c in args.cats.split(",") if c.strip()]
    pois = select_pois(args.business, cats, args.city, args.state, args.min_reviews)
    if pois.empty:
        sys.exit("Tidak ada POI yang cocok. Cek --cats/--city/--state.")
    print(f"POI terpilih: {len(pois)}")

    rv = load_reviews(args.reviews, set(pois["business_id"]), args.review_cache)
    rv = cap_reviews(rv, args.max_reviews_per_poi)
    print(f"Review dipakai: {len(rv)} (maks {args.max_reviews_per_poi} per POI)")

    qdf = pd.read_csv(args.queries)
    qids, qtexts = qdf["id"].tolist(), qdf["text"].tolist()

    mask = metro_mask(qdf, pois)
    mats = {"bm25": apply_mask(bm25_scores(pois, rv, qtexts), mask)}
    if args.dense.lower() != "none":
        enc = load_encoder(args.dense)
        emb_cache = os.path.join(args.outdir, "review_emb.npy")
        mats["dense"] = apply_mask(
            dense_scores(enc, pois, rv, qtexts, pool=args.pool, emb_cache=emb_cache), mask)
        mats["rrf"] = rrf([mats["bm25"], mats["dense"]])

    rows = []
    for name, S in mats.items():
        rows += topk_rows(qids, pois, S, name, args.topk, mask)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.outdir, "runs_text.csv"), index=False)
    pool = build_pool(runs, qdf, pois, args.pool_k)
    pool.to_csv(os.path.join(args.outdir, "pool_for_annotation.csv"), index=False)

    print(f"\nTersimpan di {args.outdir}/")
    print(f"  runs_text.csv            : top-{args.topk} per query per metode")
    print(f"  pool_for_annotation.csv  : {len(pool)} pasangan unik (top-{args.pool_k} gabungan)")


if __name__ == "__main__":
    main()
