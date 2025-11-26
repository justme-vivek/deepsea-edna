#!/usr/bin/env python3
"""
02_reduce_cluster.py

UMAP + HDBSCAN clustering of embeddings with safe memmap loading (meta fallback).

Inputs (defaults):
 - embeddings: /Users/ansh/deepsea_edna/data/DNABERT_embeddings/windows_embeddings.npy
 - meta:       /Users/ansh/deepsea_edna/data/DNABERT_embeddings/windows_embeddings_meta.npy
 - index:      /Users/ansh/deepsea_edna/data/DNABERT_embeddings/windows_index.tsv
 - fasta (optional, used to fetch sequences if index lacks them):
               /Users/ansh/deepsea_edna/data/preprocess/ref_dnabert_windows.fa

Outputs (saved under results_dir, default /Users/ansh/deepsea_edna/data/CLUSTER_files):
 - clusters.tsv
 - cluster_summary.tsv
 - cluster_reps.fa
 - umap.png

This script is robust to `.npy` pickled-object errors by using a float32 memmap when a companion meta file exists.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import csv
import sys

def safe_load_embeddings(emb_path: Path):
    """
    Try to open embeddings safely:
    - If a companion meta file exists (windows_embeddings_meta.npy) with [N, D], open a float32 memmap.
    - Otherwise try np.load(..., mmap_mode='r') and return the array (may raise on pickle).
    """
    meta_path = emb_path.parent / "windows_embeddings_meta.npy"
    if meta_path.exists():
        try:
            meta = np.load(str(meta_path))
            # support shape stored as array [N, D] or as tuple-like
            if hasattr(meta, "__len__") and len(meta) >= 2:
                N_expected = int(meta[0])
                D_expected = int(meta[1])
            else:
                raise ValueError("meta file does not contain two integers")
            emb = np.memmap(str(emb_path), dtype="float32", mode="r", shape=(N_expected, D_expected))
            print(f"[INFO] Opened memmap {emb_path} with shape {(N_expected, D_expected)}", file=sys.stderr)
            return emb
        except Exception as e:
            print(f"[WARN] Failed to open memmap using meta {meta_path}: {e}", file=sys.stderr)
            # fall through to try normal load
    # Fallback: try normal np.load (may raise ValueError if object-pickled)
    try:
        arr = np.load(str(emb_path), mmap_mode="r")
        print(f"[INFO] Loaded embeddings via np.load; shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)}", file=sys.stderr)
        return arr
    except Exception as e:
        raise SystemExit(f"Failed to load embeddings from {emb_path}. Error: {e}\n"
                         "If the file contains pickled objects, recreate the embeddings as a float32 array or provide a companion windows_embeddings_meta.npy with [N, D].")

def write_index(records, out_dir: Path):
    idx_path = out_dir / "windows_index.tsv"
    with open(idx_path, "w") as fh:
        fh.write("idx\twindow_id\theader\tsequence\n")
        for i, rec in enumerate(records):
            fh.write(f"{i}\t{rec.id}\t{rec.id}\t{str(rec.seq)}\n")
    print(f"[INFO] Wrote index TSV to {idx_path}", file=sys.stderr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="/Users/ansh/deepsea_edna/data/DNABERT_embeddings/windows_embeddings.npy",
                   help="Path to embeddings .npy (float32 memmap expected)")
    p.add_argument("--index", default="/Users/ansh/deepsea_edna/data/DNABERT_embeddings/windows_index.tsv",
                   help="Index TSV mapping idx -> window_id/header/sequence (if sequence present)")
    p.add_argument("--windows_fa", default="/Users/ansh/deepsea_edna/data/preprocess/ref_dnabert_windows.fa",
                   help="Original windows FASTA (used if index lacks sequences)")
    p.add_argument("--results_dir", default="/Users/ansh/deepsea_edna/data/CLUSTER_files",
                   help="Output folder for clusters/results")
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--umap_components", type=int, default=2)
    p.add_argument("--hdb_min_cluster_size", type=int, default=5)
    p.add_argument("--hdb_min_samples", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    emb_path = Path(args.emb)
    idx_path = Path(args.index)
    fasta_path = Path(args.windows_fa)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading embeddings...", file=sys.stderr)
    emb = safe_load_embeddings(emb_path)
    # emb may be memmap or ndarray
    try:
        N, D = emb.shape
    except Exception:
        raise SystemExit("[ERROR] Unexpected embedding shape; aborting.")

    print(f"[INFO] Embeddings shape: {N, D}", file=sys.stderr)

    print("[INFO] Loading index TSV...", file=sys.stderr)
    if not idx_path.exists():
        print(f"[WARN] Index {idx_path} not found. Will try to build index from FASTA.", file=sys.stderr)
        if not fasta_path.exists():
            raise SystemExit("Neither index TSV nor windows FASTA found; cannot proceed.")
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        write_index(records, Path(args.emb).parent)
        idx_df = pd.read_csv(idx_path, sep="\t", header=0, dtype=str)
        idx_df["idx"] = idx_df["idx"].astype(int)
    else:
        idx_df = pd.read_csv(idx_path, sep="\t", header=0, dtype=str)
        if "idx" in idx_df.columns:
            idx_df["idx"] = idx_df["idx"].astype(int)
            # ensure ordering covers 0..N-1
            idx_df = idx_df.set_index("idx").reindex(range(N)).reset_index()
        else:
            # create idx if missing
            idx_df.insert(0, "idx", range(N))
            # ensure columns for window_id/header
            if idx_df.shape[1] < 2:
                idx_df["window_id"] = idx_df.iloc[:,1].astype(str)
            if "header" not in idx_df.columns:
                idx_df["header"] = idx_df["window_id"]

    # Standardize and run UMAP/HDBSCAN
    print("[INFO] Standardizing embeddings...", file=sys.stderr)
    scaler = StandardScaler()
    # convert memmap to array for UMAP (UMAP expects array-like in memory)
    X = scaler.fit_transform(np.array(emb))

    print("[INFO] Running UMAP...", file=sys.stderr)
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        n_components=args.umap_components, random_state=args.random_state)
    X2 = reducer.fit_transform(X)
    print("[INFO] UMAP done. shape:", X2.shape, file=sys.stderr)

    print("[INFO] Running HDBSCAN...", file=sys.stderr)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.hdb_min_cluster_size, min_samples=args.hdb_min_samples)
    labels = clusterer.fit_predict(X2)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] HDBSCAN done. clusters found (excluding -1): {n_clusters}", file=sys.stderr)

    # Write clusters.tsv
    clusters_tsv = out_dir / "clusters.tsv"
    print(f"[INFO] Writing clusters to {clusters_tsv}", file=sys.stderr)
    with open(clusters_tsv, "w") as fh:
        fh.write("idx\twindow_id\theader\tcluster\n")
        for i, label in enumerate(labels):
            row = idx_df.loc[idx_df["idx"] == i]
            window_id = row["window_id"].values[0]
            header = row["header"].values[0] if "header" in row else window_id
            fh.write(f"{i}\t{window_id}\t{header}\t{label}\n")

    # Select cluster representatives (medoid in UMAP space)
    print("[INFO] Selecting cluster representatives...", file=sys.stderr)
    df = pd.DataFrame({
        "idx": range(N),
        "window_id": idx_df["window_id"].values,
        "header": idx_df["header"].values if "header" in idx_df else idx_df["window_id"].values,
        "cluster": labels
    })

    reps = []
    summary_rows = []
    for cl in sorted(set(labels)):
        if cl == -1:
            continue
        members = df[df["cluster"] == cl]
        member_idxs = members["idx"].values
        coords = X2[member_idxs]
        centroid = coords.mean(axis=0)
        dists = ((coords - centroid) ** 2).sum(axis=1)
        pick_local = int(dists.argmin())
        pick_idx = int(member_idxs[pick_local])
        rep_row = idx_df.loc[idx_df["idx"] == pick_idx].iloc[0]
        seq = rep_row["sequence"] if "sequence" in rep_row and pd.notna(rep_row["sequence"]) else None
        reps.append(SeqRecord(Seq(seq if seq is not None else ""), id=f"cluster{cl}_rep|idx{pick_idx}|{rep_row['window_id']}", description=""))
        summary_rows.append((cl, int(len(members)), pick_idx, rep_row["window_id"]))

    # If sequences missing, load fasta and populate seqs
    need_seq = any(len(r.seq) == 0 for r in reps)
    if need_seq and fasta_path.exists():
        fasta_map = {rec.id: str(rec.seq) for rec in SeqIO.parse(str(fasta_path), "fasta")}
        for i, r in enumerate(reps):
            if len(r.seq) == 0:
                parts = r.id.split("|")
                hdr = parts[-1] if len(parts) >= 3 else parts[0]
                seq = fasta_map.get(hdr)
                if seq:
                    reps[i].seq = Seq(seq)

    # Write cluster reps
    reps_fa = out_dir / "cluster_reps.fa"
    SeqIO.write(reps, str(reps_fa), "fasta")
    print(f"[INFO] Wrote cluster reps to {reps_fa}", file=sys.stderr)

    # Write summary TSV
    summary_tsv = out_dir / "cluster_summary.tsv"
    with open(summary_tsv, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["cluster", "size", "rep_idx", "rep_window_id"])
        for r in summary_rows:
            w.writerow(r)
    print(f"[INFO] Wrote cluster summary to {summary_tsv}", file=sys.stderr)

    # UMAP scatter plot
    umap_png = out_dir / "umap.png"
    plt.figure(figsize=(8, 6))
    unique = sorted(set(labels))
    # color mapping
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
    colors = [cmap(i % 20) if lab != -1 else (0.8, 0.8, 0.8) for i, lab in enumerate(labels)]
    plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=8, alpha=0.8)
    plt.title("UMAP + HDBSCAN clusters")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(str(umap_png), dpi=200)
    print(f"[INFO] Saved UMAP plot to {umap_png}", file=sys.stderr)

    print("[DONE] Results written to", out_dir, file=sys.stderr)


if __name__ == "__main__":
    main()
