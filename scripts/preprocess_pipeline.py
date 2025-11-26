#!/usr/bin/env python3
"""
ref_preprocess_nt_uncultured.py

Hardcoded preprocessing pipeline for your dataset:
 Input FASTA:  /Users/ansh/deepsea_edna/data/nt_uncultured_1000.fa
 Output dir:   /Users/ansh/deepsea_edna/data/preprocess

Produces:
 - ref_clean.fa
 - ref_metadata.tsv
 - ref_dnabert.fa
 - ref_dnabert_windows.fa
 - dnabert_labels.tsv
 - (optional) BLAST DB files if RUN_MAKEBLASTDB = True
"""

import os, re, subprocess, sys
from pathlib import Path
from textwrap import wrap

# -------------------- HARD-CODED SETTINGS --------------------
INPUT = "/Users/ansh/deepsea_edna/data/nt_uncultured_1000.fa"
OUTDIR = "/Users/ansh/deepsea_edna/data/preprocess"
WINDOW = 250       # DNABERT window length (0 = no windows)
STRIDE = 50        # sliding window stride
MINLEN = 50        # minimum sequence length
KEEP_NS = False    # keep sequences with N in ref_clean.fa?
RUN_MAKEBLASTDB = False  # set True if you want BLAST DB built automatically
# --------------------------------------------------------------

def read_fasta(path):
    name = None; seq_lines = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line: continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_lines)
                name = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if name is not None:
            yield name, "".join(seq_lines)

def clean_seq(seq):
    return re.sub(r"[^ACGTNacgtn]", "", seq).upper()

def extract_label_from_description(desc):
    m = re.search(r"([A-Z][a-z]+ [a-z]+)", desc)
    if m: return m.group(1)
    tokens = [t for t in re.split(r"\s+", desc) if t]
    if len(tokens) >= 2: return f"{tokens[0]} {tokens[1]}"
    if tokens: return tokens[0]
    return "unknown"

def wrap_and_write(fh, seq, w=80):
    for line in wrap(seq, w):
        fh.write(line + "\n")

def write_ref_clean_and_meta(input_fa, out_clean_fa, out_meta_tsv, minlen, keep_ns):
    meta_lines = []; kept = 0; total = 0
    with open(out_clean_fa, "w") as out:
        for hdr, seq in read_fasta(input_fa):
            total += 1
            seqc = clean_seq(seq)
            if len(seqc) < minlen: continue
            ncount = seqc.count("N")
            if (not keep_ns) and ncount > 0: continue
            acc = hdr.split()[0]
            desc = " ".join(hdr.split()[1:]).strip()
            label = extract_label_from_description(desc) if desc else acc
            clean_header = f"{acc} {label}"
            out.write(f">{clean_header}\n")
            wrap_and_write(out, seqc)
            meta_lines.append((acc, label, desc, clean_header, len(seqc), ncount))
            kept += 1
    with open(out_meta_tsv, "w") as mh:
        mh.write("accession\tlabel\tdescription\tclean_header\tlength\tn_count\n")
        for row in meta_lines:
            mh.write("\t".join([str(x) for x in row]) + "\n")
    print(f"Wrote {kept}/{total} cleaned sequences to {out_clean_fa}; metadata -> {out_meta_tsv}")

def build_blast_db(clean_fa, db_prefix):
    cmd = ["makeblastdb", "-in", str(clean_fa), "-dbtype", "nucl", "-parse_seqids", "-out", str(db_prefix)]
    try:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("BLAST DB built with prefix:", db_prefix)
    except FileNotFoundError:
        print("makeblastdb not found. To build manually run:\n" + " ".join(cmd))

def make_dnabert_fasta_and_windows(clean_fa, dnabert_fa, dnabert_windows_fa, labels_tsv, window, stride):
    id_to_label = {}; seqs = []
    for hdr, seq in read_fasta(clean_fa):
        acc = hdr.split()[0]
        label = " ".join(hdr.split()[1:]) if len(hdr.split())>1 else acc
        id_to_label[acc] = label
        seqs.append((acc, seq))
    written = 0
    with open(dnabert_fa, "w") as outf:
        for acc, seq in seqs:
            if re.search(r"[^ACGT]", seq): continue
            outf.write(f">{acc}\n")
            wrap_and_write(outf, seq)
            written += 1
    print(f"Wrote DNABERT-ready {written} sequences (ACGT-only) -> {dnabert_fa}")
    with open(labels_tsv, "w") as lh, open(dnabert_windows_fa, "w") as wh:
        lh.write("id\tlabel\n")
        total_windows = 0
        for acc, seq in seqs:
            if re.search(r"[^ACGT]", seq): continue
            if len(seq) < window:
                sid = f"{acc}_full"
                wh.write(f">{sid}\n"); wrap_and_write(wh, seq)
                lh.write(f"{sid}\t{id_to_label.get(acc, 'unknown')}\n")
                total_windows += 1
                continue
            for i in range(0, len(seq)-window+1, stride):
                chunk = seq[i:i+window]
                wid = f"{acc}_w{i}"
                wh.write(f">{wid}\n"); wrap_and_write(wh, chunk)
                lh.write(f"{wid}\t{id_to_label.get(acc, 'unknown')}\n")
                total_windows += 1
            last_start = len(seq) - window
            if last_start % stride != 0 and last_start > 0:
                chunk = seq[last_start:last_start+window]
                wid = f"{acc}_w{last_start}"
                wh.write(f">{wid}\n"); wrap_and_write(wh, chunk)
                lh.write(f"{wid}\t{id_to_label.get(acc, 'unknown')}\n")
                total_windows += 1
        print(f"Wrote {total_windows} DNABERT windows -> {dnabert_windows_fa} and labels -> {labels_tsv}")

def main():
    inp = Path(INPUT); outdir = Path(OUTDIR)
    if not inp.exists():
        print("Input FASTA not found:", inp); sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)
    clean_fa = outdir / "ref_clean.fa"
    meta_tsv = outdir / "ref_metadata.tsv"
    db_prefix = outdir / "my_ref_db"
    dnabert_fa = outdir / "ref_dnabert.fa"
    dnabert_windows_fa = outdir / "ref_dnabert_windows.fa"
    labels_tsv = outdir / "dnabert_labels.tsv"

    write_ref_clean_and_meta(inp, clean_fa, meta_tsv, MINLEN, KEEP_NS)
    if RUN_MAKEBLASTDB:
        build_blast_db(clean_fa, db_prefix)
    else:
        print("(makeblastdb not run; enable RUN_MAKEBLASTDB=True to build automatically)")

    make_dnabert_fasta_and_windows(clean_fa, dnabert_fa, dnabert_windows_fa, labels_tsv, WINDOW, STRIDE)
    print("\nDone. Outputs written to:", outdir)

if __name__ == "__main__":
    main()
