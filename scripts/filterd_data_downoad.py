#!/usr/bin/env python3
"""
Stream nt.gz from NCBI, filter "uncultured/environmental/clone/metagenome" headers,
prompt for N, and write exactly N matched full FASTA records to:
    /Users/ansh/deepsea_edna/data/nt_uncultured_<N>.fa

Usage:
    python scripts/filter_stream_prompt_nt.py

Dependencies:
    pip install requests
"""

import sys
import time
import re
import zlib
from pathlib import Path

# Try to import requests; helpful message if missing
try:
    import requests
except Exception:
    print("Error: the 'requests' package is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


# ----- Config -----
DATA_DIR = Path("/Users/ansh/deepsea_edna/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Keywords (regex fragments) to recognize "uncultured/environmental" headers
KEYWORD_FRAGMENTS = [
    r"uncultur",      # uncultured, uncultured eukaryote
    r"environmental", # environmental sample
    r"metagenom",     # metagenome, metagenomic
    r"clone",         # clone entries
    r"env[-_ ]",      # env- or env_
    r"unidentified",
    r"novel",
    r"symbiont",
]
KEYWORD_RE = re.compile("|".join(KEYWORD_FRAGMENTS), flags=re.IGNORECASE)


def prompt_for_n(default=1000):
    """Prompt user for integer N, return default for empty input."""
    while True:
        try:
            s = input(f"How many filtered FASTA records do you want? [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled. Exiting.")
            sys.exit(1)
        if s == "":
            return default
        try:
            v = int(s)
            if v <= 0:
                print("Please enter a positive integer.")
                continue
            return v
        except ValueError:
            print("Please enter a valid integer (or press Enter for default).")


def print_progress(compr_bytes, decomp_bytes, matches):
    """Single-line progress updater."""
    compr_mb = compr_bytes / (1024 * 1024)
    decomp_mb = decomp_bytes / (1024 * 1024)
    sys.stdout.write(f"\rCompressed: {compr_mb:7.2f} MB | Decompressed: {decomp_mb:7.2f} MB | Matches: {matches}")
    sys.stdout.flush()


def stream_filter_nt_write_n(out_path: Path, n_matches: int, chunk_size=65536):
    """
    Stream nt.gz, filter headers by KEYWORD_RE, write EXACTLY n_matches complete FASTA records
    to out_path. Returns number written.
    """
    url = "https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/nt.gz"
    decomp = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)

    compressed_total = 0
    decompressed_total = 0
    last_progress = time.time()

    # State
    buffer = ""               # holds decompressed partial text
    writing_current = False   # whether current record is being written to file
    matches_found = 0
    stop_after_current = False  # when True, stop when next header appears (we've completed N)
    any_written = False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with out_path.open("w", encoding="utf-8") as fout:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_len = r.headers.get("content-length")
                if total_len:
                    print(f"Streaming {url}  (size ≈ {int(total_len)/1e6:.2f} MB)")
                else:
                    print(f"Streaming {url}")

                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    compressed_total += len(chunk)
                    # decompress
                    try:
                        dec_bytes = decomp.decompress(chunk)
                    except Exception:
                        # safe fallback
                        dec_bytes = decomp.decompress(chunk, max_length=0)
                    decompressed_total += len(dec_bytes)

                    # decode
                    try:
                        text = dec_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        text = dec_bytes.decode("latin1", errors="replace")

                    buffer += text

                    # Process full lines; keep trailing partial line in buffer
                    if "\n" in buffer:
                        parts = buffer.split("\n")
                        buffer = parts.pop()  # leftover partial line

                        for line in parts:
                            # New header
                            if line.startswith(">"):
                                # If we previously hit N and are set to stop after current, stop now
                                if stop_after_current:
                                    if any_written:
                                        # final progress & return
                                        print()
                                        print(f"Reached target of {n_matches} matched records — stopping stream.")
                                    return matches_found

                                # Evaluate header for keyword match
                                if KEYWORD_RE.search(line):
                                    # This header should be written (a matched record)
                                    matches_found += 1
                                    writing_current = True
                                    any_written = True
                                    fout.write(line + "\n")
                                    # If we've just reached N matches, set stop flag (we'll finish this record fully,
                                    # and then stop when next header appears)
                                    if matches_found >= n_matches:
                                        stop_after_current = True
                                else:
                                    # header doesn't match: only write it if currently writing (i.e., if previous record matched)
                                    writing_current = False
                                    # do not write this header
                            else:
                                # sequence or metadata line
                                if writing_current:
                                    fout.write(line + "\n")

                    # progress update every 0.25s
                    now = time.time()
                    if now - last_progress >= 0.25:
                        print_progress(compressed_total, decompressed_total, matches_found)
                        last_progress = now

                # Stream ended -> flush decompressor tail
                try:
                    tail_bytes = decomp.flush()
                except Exception:
                    tail_bytes = b""
                decompressed_total += len(tail_bytes)
                try:
                    tail_text = tail_bytes.decode("utf-8", errors="replace")
                except Exception:
                    tail_text = tail_bytes.decode("latin1", errors="replace")

                buffer += tail_text

                # process remaining lines in buffer
                if buffer:
                    parts = buffer.split("\n")
                    for line in parts:
                        if line == "":
                            continue
                        if line.startswith(">"):
                            if stop_after_current:
                                # we've finished required matches earlier, return
                                print_progress(compressed_total, decompressed_total, matches_found)
                                print()
                                print(f"Reached target of {n_matches} matched records in tail — done.")
                                return matches_found
                            if KEYWORD_RE.search(line):
                                matches_found += 1
                                writing_current = True
                                any_written = True
                                fout.write(line + "\n")
                                if matches_found >= n_matches:
                                    stop_after_current = True
                            else:
                                writing_current = False
                        else:
                            if writing_current:
                                fout.write(line + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial output may exist.")
        return matches_found

    # Final progress
    print_progress(compressed_total, decompressed_total, matches_found)
    print()
    print(f"Stream ended. Wrote {matches_found} matched records to {out_path}")
    return matches_found


def main():
    print("This will stream: https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/nt.gz")
    n = prompt_for_n(default=1000)
    out_path = DATA_DIR / f"nt_uncultured_{n}.fa"
    print(f"Filtered output will be saved to: {out_path}")
    written = stream_filter_nt_write_n(out_path, n_matches=n)
    print(f"Finished: wrote {written} matched records to {out_path}")


if __name__ == "__main__":
    main()
