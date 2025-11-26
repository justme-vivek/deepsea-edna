#!/usr/bin/env python3
"""
01_embed_dnabert6.py

Generate embeddings from the original DNABERT (6-mer) model (zhihan1996/DNA_bert_6).

Outputs (out_dir):
 - windows_embeddings.npy      (N x D float32 memmap)
 - windows_embeddings_meta.npy (N, D)
 - windows_index.tsv           (idx<TAB>window_id<TAB>header<TAB>sequence)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import torch
from Bio import SeqIO

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Missing transformers. Install with: pip install transformers", file=sys.stderr)
    sys.exit(1)


def choose_device(prefer_mps=True):
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_index(records, out_dir):
    index_path = out_dir / "windows_index.tsv"
    with open(index_path, "w") as fh:
        fh.write("idx\twindow_id\theader\tsequence\n")
        for i, rec in enumerate(records):
            fh.write(f"{i}\t{rec.id}\t{rec.id}\t{str(rec.seq)}\n")
    print(f"[INFO] Wrote index TSV to {index_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_fa", default="/Users/ansh/deepsea_edna/data/preprocess/ref_dnabert_windows.fa",
                   help="Input FASTA with windows (A/C/G/T only)")
    p.add_argument("--out_dir", default="/Users/ansh/deepsea_edna/data/DNABERT_embeddings",
                   help="Output directory")
    p.add_argument("--model_name", default="zhihan1996/DNA_bert_6",
                   help="Public DNABERT (6-mer) model")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = p.parse_args()

    device = choose_device(prefer_mps=True) if args.device == "auto" else torch.device(args.device)
    print(f"[INFO] Selected device: {device}", file=sys.stderr)

    windows_fa = Path(args.windows_fa)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not windows_fa.exists():
        sys.exit(f"Input FASTA not found: {windows_fa}")

    records = list(SeqIO.parse(str(windows_fa), "fasta"))
    if not records:
        sys.exit("No sequences found in input FASTA.")
    N = len(records)
    print(f"[INFO] Loaded {N} windows from {windows_fa}", file=sys.stderr)

    print(f"[INFO] Loading tokenizer/model: {args.model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device).eval()

    # check hidden dimension
    with torch.no_grad():
        test_seq = str(records[0].seq)
        enc = tokenizer([test_seq], return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        H = out.last_hidden_state.shape[-1]
    print(f"[INFO] Model hidden dim = {H}", file=sys.stderr)

    emb_path = out_dir / "windows_embeddings.npy"
    emb_mem = np.memmap(str(emb_path), dtype="float32", mode="w+", shape=(N, H))

    bs = args.batch_size
    idx = 0
    with torch.no_grad():
        for start in range(0, N, bs):
            end = min(N, start + bs)
            seqs = [str(r.seq) for r in records[start:end]]
            enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last_hidden = out.last_hidden_state  # (B,L,H)
            attn = enc.get("attention_mask")
            if attn is None:
                pooled = last_hidden.mean(dim=1)
            else:
                attn = attn.unsqueeze(-1)
                pooled = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
            emb_mem[start:end, :] = pooled.cpu().numpy().astype("float32", copy=False)
            idx += len(seqs)
            print(f"[INFO] Processed {idx}/{N} windows", file=sys.stderr)

    emb_mem.flush()
    np.save(out_dir / "windows_embeddings_meta.npy", np.array([N, H], dtype=np.int64))
    print(f"[DONE] Saved embeddings to {emb_path} (shape {N},{H})", file=sys.stderr)

    write_index(records, out_dir)


if __name__ == "__main__":
    main()
