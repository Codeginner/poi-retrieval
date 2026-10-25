#!/usr/bin/env python3
"""
Audit Yelp Open Dataset untuk milih kota + kategori benchmark POI retrieval.

Tahap 1 - audit (cepat, tanpa baca review.json yang besar):
    python audit_yelp.py \
        --business /path/yelp_academic_dataset_business.json \
        --photos   /path/photos.json \
        --cats "Coffee & Tea,Cafes" --min-reviews 20

    Opsi bantu: --list-cats  (lihat 40 kategori terbanyak, buat cek nama kategori)

Tahap 2 - setelah milih kota, dump subset (baca review.json, butuh beberapa menit):
    python audit_yelp.py ...(argumen yang sama)... \
        --city Nashville --state TN \
        --reviews /path/yelp_academic_dataset_review.json \
        --photo-dir /path/photos \
        --dump subset_city

Hasil dump (business.jsonl, photos.jsonl, reviews.jsonl, photos/) bisa di-upload
sebagai Kaggle dataset kecil, jadi file Yelp mentah yang besar tidak perlu dipakai lagi.

Catatan:
- Semua file Yelp berformat JSON-lines (satu objek per baris).
- Nama field mengikuti dokumentasi Yelp (business_id, categories, attributes, hours,
  review_count, city, state, photo_id, label). Cek kalau versi dataset lu beda.
- Atribut kompleks seperti Ambience tersimpan sebagai string, bukan dict; parse nanti
  dengan ast.literal_eval.
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

PHOTO_LABELS = ["inside", "outside", "food", "drink", "menu"]


def read_jsonl(path, **kw):
    return pd.read_json(path, lines=True, dtype=False, convert_dates=False, **kw)


def load_business(path):
    df = read_jsonl(path)
    df["categories"] = df["categories"].fillna("")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    return df


def filter_categories(df, cats):
    wanted = {c.strip().lower() for c in cats if c.strip()}

    def match(s):
        return any(c.strip().lower() in wanted for c in s.split(","))

    out = df[df["categories"].map(match)].copy()
    for col in ("hours", "attributes"):  # jaga-jaga kalau versi dataset tidak punya kolomnya
        if col not in out.columns:
            out[col] = None
    out["has_hours"] = out["hours"].map(lambda x: isinstance(x, dict) and len(x) > 0)
    out["has_attr"] = out["attributes"].map(lambda x: isinstance(x, dict) and len(x) > 0)
    return out


def photo_table(photos, ids):
    p = photos[photos["business_id"].isin(ids)]
    t = p.pivot_table(index="business_id", columns="label", values="photo_id",
                      aggfunc="count", fill_value=0)
    for lab in PHOTO_LABELS:
        if lab not in t.columns:
            t[lab] = 0
    t = t[PHOTO_LABELS].copy()
    t["n_photo"] = t.sum(axis=1)
    t.columns = [f"ph_{c}" if c != "n_photo" else c for c in t.columns]
    return t


def add_photo_counts(df, photos):
    t = photo_table(photos, set(df["business_id"]))
    df = df.merge(t, left_on="business_id", right_index=True, how="left")
    for c in t.columns:
        df[c] = df[c].fillna(0).astype(int)
    return df


def city_table(df, have_photos):
    g = df.groupby(["state", "city"])
    t = g.agg(n_biz=("business_id", "size"),
              n_cand=("cand", "sum"),
              med_reviews=("review_count", "median"),
              pct_hours=("has_hours", "mean"),
              pct_attr=("has_attr", "mean"))
    if have_photos:
        t["pct_inside"] = g["ph_inside"].apply(lambda s: (s > 0).mean())
        t["pct_outside"] = g["ph_outside"].apply(lambda s: (s > 0).mean())
    t = t.sort_values("n_cand", ascending=False)
    return t.round(2)


def attr_coverage(df, top=20):
    n = len(df)
    counts = {}
    for a in df["attributes"]:
        if isinstance(a, dict):
            for k, v in a.items():
                if v not in (None, "None"):
                    counts[k] = counts.get(k, 0) + 1
    s = pd.Series(counts, dtype=float).sort_values(ascending=False).head(top) / max(n, 1)
    return s.round(2)


def dump_subset(df_city, photos, reviews_path, outdir, photo_dir, max_per_label):
    os.makedirs(outdir, exist_ok=True)
    ids = set(df_city["business_id"])

    df_city.to_json(os.path.join(outdir, "business.jsonl"),
                    orient="records", lines=True, force_ascii=False)
    print(f"business.jsonl : {len(df_city)} bisnis")

    if photos is not None:
        p = photos[photos["business_id"].isin(ids)]
        p.to_json(os.path.join(outdir, "photos.jsonl"),
                  orient="records", lines=True, force_ascii=False)
        print(f"photos.jsonl   : {len(p)} baris metadata")
        if photo_dir:
            os.makedirs(os.path.join(outdir, "photos"), exist_ok=True)
            copied = missing = 0
            for lab in ("inside", "outside"):
                sub = p[p["label"] == lab].groupby("business_id").head(max_per_label)
                for pid in sub["photo_id"]:
                    src = os.path.join(photo_dir, f"{pid}.jpg")
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(outdir, "photos", f"{pid}.jpg"))
                        copied += 1
                    else:
                        missing += 1
            print(f"foto dicopy    : {copied} (hilang: {missing}), "
                  f"maks {max_per_label} per label per bisnis")

    if reviews_path:
        n_rev, n_chars = 0, 0
        with open(os.path.join(outdir, "reviews.jsonl"), "w", encoding="utf-8") as f:
            reader = read_jsonl(reviews_path, chunksize=200_000)
            for ch in reader:
                ch = ch[ch["business_id"].isin(ids)]
                if ch.empty:
                    continue
                n_rev += len(ch)
                n_chars += int(ch["text"].str.len().sum())
                s = ch.to_json(orient="records", lines=True, force_ascii=False)
                f.write(s if s.endswith("\n") else s + "\n")
        print(f"reviews.jsonl  : {n_rev} review, rata-rata {n_chars / max(n_rev, 1):.0f} karakter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business", required=True)
    ap.add_argument("--photos", help="photos.json (metadata foto, field: photo_id, business_id, label)")
    ap.add_argument("--reviews", help="review.json, hanya dipakai saat --dump")
    ap.add_argument("--cats", default="Coffee & Tea,Cafes", help="kategori Yelp, pisah koma")
    ap.add_argument("--min-reviews", type=int, default=20)
    ap.add_argument("--need-labels", default="inside,outside",
                    help="label foto yang wajib ada utk jadi kandidat, pisah koma "
                         "(pantai/taman: pakai 'outside'; kosongkan dengan '' untuk tanpa syarat foto)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--list-cats", action="store_true")
    ap.add_argument("--city", nargs="+", help="nama kota; boleh lebih dari satu kata, mis. --city New Orleans")
    ap.add_argument("--state")
    ap.add_argument("--dump", help="folder output subset kota terpilih")
    ap.add_argument("--photo-dir", help="folder berisi <photo_id>.jpg")
    ap.add_argument("--max-photos-per-label", type=int, default=5)
    ap.add_argument("--out", default="yelp_audit_city.csv")
    args = ap.parse_args()
    args.city = " ".join(args.city) if args.city else None

    raw = load_business(args.business)
    print(f"Total bisnis di file: {len(raw)}")

    if args.list_cats:
        cats = raw["categories"].str.split(",").explode().str.strip()
        print(cats.value_counts().head(40).to_string())
        return

    cats = [c.strip() for c in args.cats.split(",")]
    df = filter_categories(raw, cats)
    if df.empty:
        sys.exit("Tidak ada bisnis yang cocok. Cek nama kategori dengan --list-cats.")
    print(f"Bisnis kategori {cats}: {len(df)}")

    photos = read_jsonl(args.photos) if args.photos else None
    have_photos = photos is not None
    if have_photos:
        df = add_photo_counts(df, photos)
        need = [x.strip() for x in args.need_labels.split(",") if x.strip()]
        bad = [x for x in need if x not in PHOTO_LABELS]
        if bad:
            sys.exit(f"Label tidak dikenal: {bad}. Pilihan: {PHOTO_LABELS}")
        df["cand"] = df["review_count"] >= args.min_reviews
        for lab in need:
            df["cand"] &= df[f"ph_{lab}"] > 0
    else:
        df["cand"] = df["review_count"] >= args.min_reviews
        print("(tanpa --photos: kandidat hanya dihitung dari jumlah review)")

    t = city_table(df, have_photos)
    t.to_csv(args.out)
    print(f"\nTop {args.top} kota (urut jumlah kandidat; kandidat = review >= "
          f"{args.min_reviews}" + (f" dan punya foto {args.need_labels or '(any)'}" if have_photos else "") + "):")
    print(t.head(args.top).to_string())
    print(f"\nTabel lengkap tersimpan: {args.out}")

    if not args.city:
        print("\nLangkah berikut: pilih kota, lalu jalankan lagi dengan --city dan --state.")
        return

    sel = df[df["city"].str.lower() == args.city.lower()]
    if args.state:
        sel = sel[sel["state"].str.lower() == args.state.lower()]
    if sel.empty:
        sys.exit("Kota tidak ditemukan. Cek ejaan di tabel di atas.")

    pool = sel[sel["review_count"] >= args.min_reviews]
    ncand = int(sel["cand"].sum())
    print(f"\n== {args.city} ==")
    print(f"Bisnis kategori ini : {len(sel)}")
    print(f"Review >= {args.min_reviews}        : {len(pool)}")
    print(f"Kandidat            : {ncand}")
    print("\nCoverage atribut (fraksi bisnis yang punya nilai):")
    print(attr_coverage(pool).to_string())
    # Patokan kasar dari gw, bukan angka baku.
    print("\nHint go/no-go (patokan kasar):")
    print(f"  [{'OK' if ncand >= 300 else 'KURANG'}] kandidat >= 300 (sekarang {ncand})")
    if have_photos:
        print("  Cek juga ph_inside dan ph_outside di business.jsonl hasil dump.")

    if args.dump:
        print(f"\nDump subset ke: {args.dump}")
        dump_subset(pool, photos, args.reviews, args.dump,
                    args.photo_dir, args.max_photos_per_label)


if __name__ == "__main__":
    main()
