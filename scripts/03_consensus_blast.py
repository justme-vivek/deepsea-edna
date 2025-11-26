#!/usr/bin/env python3
"""
03_consensus_blast.py

Build consensus sequences per cluster (MAFFT + majority-rule consensus) and BLAST them
against NCBI remotely (blastn -remote). Local BLAST DB usage is removed — this script
always uses NCBI remote BLAST.

Inputs (defaults):
 - clusters_tsv: /Users/ansh/deepsea_edna/data/CLUSTER_files/clusters.tsv
 - windows_fa:   /Users/ansh/deepsea_edna/data/preprocess/ref_dnabert_windows.fa
 - out_dir:      /Users/ansh/deepsea_edna/data/BLAST_files

Outputs (in out_dir):
 - cluster_members/cluster_{id}_members.fa
 - cluster_aligns/cluster_{id}.aln.fa
 - cluster_consensus.fa
 - blast_consensus_raw.tsv         (outfmt 6)
 - blast_consensus_annotated.tsv
 - blast_consensus_novel.tsv

Requirements:
 - mafft on PATH
 - blastn (BLAST+ client) installed to call `blastn -remote`
"""

import argparse
from pathlib import Path
import subprocess
import shutil
import sys
from collections import Counter, defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

def find_prog(name):
    return shutil.which(name)

def make_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_clusters(clusters_tsv):
    df = pd.read_csv(clusters_tsv, sep="\t", header=0, dtype=str)
    if "cluster" not in df.columns:
        raise SystemExit("clusters.tsv must contain 'cluster' column.")
    df["cluster"] = df["cluster"].astype(int)
    return df

def load_windows_fasta(fasta_path):
    return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(str(fasta_path), "fasta")}

def write_fasta_records(records, path):
    SeqIO.write(records, str(path), "fasta")

def run_mafft(in_fa, out_aln, mafft_prog="mafft"):
    cmd = [mafft_prog, "--auto", str(in_fa)]
    with open(out_aln, "w") as outf:
        subprocess.check_call(cmd, stdout=outf)
    return out_aln

def simple_consensus_from_alignment(aln_path, min_fraction=0.5):
    records = list(SeqIO.parse(str(aln_path), "fasta"))
    if not records:
        return ""
    L = len(records[0].seq)
    consensus_chars = []
    for pos in range(L):
        col = [r.seq[pos].upper() for r in records if r.seq[pos] != '-']
        if not col:
            consensus_chars.append('N')
            continue
        counts = Counter(col)
        base, cnt = counts.most_common(1)[0]
        if cnt / len(col) >= min_fraction and base in "ACGT":
            consensus_chars.append(base)
        else:
            consensus_chars.append('N')
    cons = "".join(consensus_chars).strip('N')
    if len(cons) == 0:
        longest = max(records, key=lambda r: len(str(r.seq).replace('-','')))
        return str(longest.seq).replace('-', '')
    return cons

def run_blast_remote_only(query_fa, out_raw, top_hits=5, task="megablast"):
    blastn = find_prog("blastn")
    if not blastn:
        raise SystemExit(
            "ERROR: blastn (BLAST+ client) not found. Install via:\n"
            "  brew install blast   (on macOS)\n"
            "or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/\n"
        )
    cmd = [
        blastn, "-query", str(query_fa),
        "-db", "nt", "-remote", "-task", task,
        "-outfmt", "6 qseqid sseqid pident length evalue bitscore stitle sacc",
        "-max_target_seqs", str(top_hits),
        "-out", str(out_raw)
    ]
    print("[INFO] Running remote BLAST (NCBI):", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)
    return out_raw

def parse_blast_and_classify(raw_tsv, out_annot, out_novel, min_pct_id=90.0, min_aln_len=100):
    if not Path(raw_tsv).exists():
        pd.DataFrame().to_csv(out_annot, sep="\t", index=False)
        pd.DataFrame().to_csv(out_novel, sep="\t", index=False)
        return
    df = pd.read_csv(raw_tsv, sep="\t", header=None,
                     names=["qseqid","sseqid","pident","length","evalue","bitscore","stitle","sacc"])
    if df.empty:
        pd.DataFrame().to_csv(out_annot, sep="\t", index=False)
        pd.DataFrame().to_csv(out_novel, sep="\t", index=False)
        return
    df_sorted = df.sort_values(["qseqid","bitscore","evalue"], ascending=[True, False, True])
    df_best = df_sorted.groupby("qseqid", as_index=False).first()
    def classify(r):
        try:
            if float(r.pident) >= float(min_pct_id) and int(r.length) >= int(min_aln_len):
                return "reliable"
            else:
                return "weak_or_novel"
        except:
            return "weak_or_novel"
    df_best["classification"] = df_best.apply(classify, axis=1)
    df_best.to_csv(out_annot, sep="\t", index=False)
    df_best[df_best["classification"] != "reliable"].to_csv(out_novel, sep="\t", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", default="/Users/ansh/deepsea_edna/data/CLUSTER_files/clusters.tsv")
    p.add_argument("--windows_fa", default="/Users/ansh/deepsea_edna/data/preprocess/ref_dnabert_windows.fa")
    p.add_argument("--out_dir", default="/Users/ansh/deepsea_edna/data/BLAST_files")
    p.add_argument("--min_fraction", type=float, default=0.5)
    p.add_argument("--top_hits", type=int, default=5)
    p.add_argument("--min_pct_id", type=float, default=90.0)
    p.add_argument("--min_aln_len", type=int, default=100)
    p.add_argument("--task", type=str, default="megablast")
    p.add_argument("--mafft", default="mafft")
    args = p.parse_args()

    out_dir = make_dir(args.out_dir)
    members_dir = make_dir(out_dir / "cluster_members")
    aligns_dir = make_dir(out_dir / "cluster_aligns")
    consensus_fa = out_dir / "cluster_consensus.fa"

    df = read_clusters(args.clusters)
    seq_map = load_windows_fasta(args.windows_fa)

    clusters = defaultdict(list)
    for _, row in df.iterrows():
        cl = int(row["cluster"])
        if cl == -1:
            continue
        idx = row["idx"]
        window_id = row["window_id"]
        seq = seq_map.get(window_id) or seq_map.get(row.get("header", window_id))
        if seq:
            clusters[cl].append((idx, window_id, seq))
        else:
            print(f"[WARN] Sequence for {window_id} not found in FASTA", file=sys.stderr)

    print(f"[INFO] Found {len(clusters)} clusters to process", file=sys.stderr)
    consensus_records = []

    mafft_prog = find_prog(args.mafft)
    for cl, members in sorted(clusters.items()):
        if not members:
            continue
        members_fa = members_dir / f"cluster_{cl}_members.fa"
        recs = [SeqRecord(Seq(seq), id=f"{idx}|{wid}", description="") for idx, wid, seq in members]
        write_fasta_records(recs, members_fa)

        if mafft_prog:
            aln_out = aligns_dir / f"cluster_{cl}.aln.fa"
            try:
                run_mafft(members_fa, aln_out, mafft_prog)
                consensus_seq = simple_consensus_from_alignment(aln_out, min_fraction=args.min_fraction)
            except subprocess.CalledProcessError:
                print(f"[WARN] MAFFT failed for cluster {cl}, using majority sequence", file=sys.stderr)
                seqs = [s for _,_,s in members]
                consensus_seq = Counter(seqs).most_common(1)[0][0]
        else:
            print("[WARN] mafft not found; using majority sequence only", file=sys.stderr)
            seqs = [s for _,_,s in members]
            consensus_seq = Counter(seqs).most_common(1)[0][0]

        if len(consensus_seq) < 50:
            consensus_seq = max(members, key=lambda x: len(x[2]))[2]

        rec_id = f"cluster{cl}_cons|n{len(members)}"
        consensus_records.append(SeqRecord(Seq(consensus_seq), id=rec_id,
                                           description=f"cluster={cl};members={len(members)}"))

    write_fasta_records(consensus_records, consensus_fa)
    print(f"[INFO] Wrote consensus FASTA: {consensus_fa}", file=sys.stderr)

    raw_out = out_dir / "blast_consensus_raw.tsv"
    run_blast_remote_only(consensus_fa, raw_out, top_hits=args.top_hits, task=args.task)

    annotated = out_dir / "blast_consensus_annotated.tsv"
    novel = out_dir / "blast_consensus_novel.tsv"
    parse_blast_and_classify(raw_out, annotated, novel,
                             min_pct_id=args.min_pct_id, min_aln_len=args.min_aln_len)

    print("[DONE] Consensus build + remote BLAST finished", file=sys.stderr)
    print("Consensus FASTA:", consensus_fa)
    print("Raw BLAST:", raw_out)
    print("Annotated:", annotated)
    print("Novel:", novel)

if __name__ == "__main__":
    main()
