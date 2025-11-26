#!/usr/bin/env python3
"""
Streamlit app to visualise DeepSea eDNA results.
Reads files from data/CLUSTER_files, data/BLAST_files, data/DNABERT_embeddings, data/preprocess.
Run: streamlit run app.py

This version:
 - preserves all original functionality (UMAP/PCA fallback, BLAST tables, rep FASTA download, validation, exports)
 - tidies UI
 - adds a sidebar toggle to render the Botpress chat inline (right column) or floating (bottom-right)
 - supports lazy-load for inline chat (loads iframe only when user clicks)
"""
from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os, io, json, re
from zipfile import ZipFile
from pathlib import Path
from Bio import SeqIO
import plotly.express as px
from sklearn.decomposition import PCA

# ----------------------------
# Config: adjust these paths if needed
# ----------------------------
BASE = Path("data")
CLUSTER_DIR = BASE / "CLUSTER_files"
BLAST_DIR = BASE / "BLAST_files"
EMB_DIR = BASE / "DNABERT_embeddings"
PRE_DIR = BASE / "preprocess"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

CLUSTERS_TSV = CLUSTER_DIR / "clusters.tsv"
CLUSTER_SUM = CLUSTER_DIR / "cluster_summary.tsv"
REPS_FASTA = CLUSTER_DIR / "cluster_reps.fa"
BLAST_ANNOT = BLAST_DIR / "blast_consensus_annotated.tsv"
BLAST_NOVEL = BLAST_DIR / "blast_consensus_novel.tsv"
EMB_NPY = EMB_DIR / "windows_embeddings.npy"
EMB_INDEX = EMB_DIR / "windows_index.tsv"
REF_META = PRE_DIR / "ref_metadata.tsv"
VALIDATION_DB = OUT_DIR / "curator_validations.json"

# ----------------------------
# Botpress shareable URL (your provided credentials)
# ----------------------------
BP_SHARE_URL = (
    "https://cdn.botpress.cloud/webchat/v3.3/shareable.html"
    "?configUrl=https://files.bpcontent.cloud/2025/09/25/20/20250925204835-XOTSN18T.json"
)

# ----------------------------
# Streamlit page config & CSS
# ----------------------------
st.set_page_config(
    layout="wide",
    page_title="JEEV • DeepSea eDNA",
    initial_sidebar_state="expanded"
)

# Minimal styling to tidy UI
st.markdown(
    """
    <style>
      .stSidebar .stButton, .stSidebar .stSelectbox { margin-top: 6px; }
      .metric { font-weight:600; }
      .card { padding:10px; border-radius:8px; background: #ffffff; box-shadow: 0 6px 18px rgba(0,0,0,0.04); }
      .small-muted { color: #6b6b6b; font-size:12px; }
      .top-title { font-size:20px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Utility helpers
# ----------------------------
def safe_read_tsv(p: Path) -> pd.DataFrame:
    if p.exists():
        try:
            return pd.read_csv(p, sep="\t")
        except Exception:
            try:
                return pd.read_csv(p)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()

def read_reps(path: Path):
    if not path.exists():
        return []
    return list(SeqIO.parse(str(path), "fasta"))

def load_embeddings(npypath: Path, indexpath: Path | None = None):
    if not npypath.exists():
        return None, None
    emb = None
    try:
        emb = np.load(npypath, allow_pickle=False)
    except Exception:
        try:
            emb_obj = np.load(npypath, allow_pickle=True)
            if isinstance(emb_obj, np.ndarray) and emb_obj.dtype == object:
                try:
                    emb = np.vstack([np.asarray(x) for x in emb_obj])
                except Exception:
                    emb = np.asarray(emb_obj.tolist())
            else:
                emb = emb_obj
        except Exception as e:
            st.warning(f"Could not load embeddings file {npypath}: {e}")
            return None, None
    if emb is None:
        return None, None
    if emb.dtype == object and emb.ndim == 1:
        try:
            emb = np.vstack([np.asarray(x) for x in emb])
        except Exception:
            st.warning("Embeddings file contains non-numeric objects and cannot be converted.")
            return None, None
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    idx = None
    if indexpath and indexpath.exists():
        try:
            idx = pd.read_csv(indexpath, sep="\t", header=None, names=["key"])
        except Exception:
            try:
                idx = pd.read_csv(indexpath)
            except Exception:
                idx = None
    return emb, idx

def save_validation(cluster_id, note="", user="local"):
    data = {}
    if VALIDATION_DB.exists():
        try:
            data = json.loads(VALIDATION_DB.read_text())
        except Exception:
            data = {}
    data[str(cluster_id)] = {"user": user, "note": note}
    VALIDATION_DB.write_text(json.dumps(data, indent=2))

## ----------------------------
# Load data
# ----------------------------
clusters_df = safe_read_tsv(CLUSTERS_TSV)
summary_df  = safe_read_tsv(CLUSTER_SUM)
blast_annot = safe_read_tsv(BLAST_ANNOT)
blast_novel = safe_read_tsv(BLAST_NOVEL)
reps = read_reps(REPS_FASTA)
emb, emb_index = load_embeddings(EMB_NPY, EMB_INDEX)

# Simple derived stats
samples_processed = clusters_df['sample'].nunique() if 'sample' in clusters_df.columns else (0)
asvs_found = clusters_df.shape[0] if not clusters_df.empty else len(reps)
clusters_found = summary_df.shape[0] if not summary_df.empty else clusters_df['cluster'].nunique() if 'cluster' in clusters_df.columns else 0
novel_candidates = blast_novel.shape[0] if not blast_novel.empty else 0

# ----------------------------
# Sidebar controls (including chat rendering toggles)
# ----------------------------
st.sidebar.header("Controls & Filters")
sel_samples = []
if 'sample' in clusters_df.columns:
    sample_list = sorted(clusters_df['sample'].unique().tolist())
    sel_samples = st.sidebar.multiselect("Select samples", options=sample_list, default=sample_list[:5] if sample_list else [])
min_cluster_size = st.sidebar.number_input("Min cluster size", min_value=1, max_value=1000, value=2)
novelty_filter = st.sidebar.selectbox("Show", options=["All", "Known only", "Novel only"], index=0)

# Chat controls
enable_chat = st.sidebar.checkbox("Enable assistant chat", value=True)
render_chat_inline = st.sidebar.checkbox("Render chat inline (right column)", value=False)
inline_lazy_load = st.sidebar.checkbox("Lazy-load inline chat (click to open)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("JEEV • DeepSea analytics — interactive report for eDNA clusters and BLAST annotation.")

# ----------------------------
# Top bar badges
# ----------------------------
col1, col2, col3, col4, col5 = st.columns([2,1,1,1,1])
with col1:
    st.markdown("##  JEEV : DEEP SEA ANALYTICS ")
    st.markdown("**Voyage ID:** VPY1234  •  **Date:** 2025-09-24")
with col2:
    st.metric("Samples Processed", samples_processed)
with col3:
    st.metric("ASVs Found", asvs_found)
with col4:
    st.metric("Clusters", clusters_found)
with col5:
    st.metric("Novel Candidates", novel_candidates)

st.markdown("---")

# ----------------------------
# Main layout: left (filters), center (visuals), right (inspector)
# ----------------------------
left_col, main_col, right_col = st.columns([1.0, 2.6, 1.0])

# LEFT: quick info + legend
with left_col:
    st.markdown("### Filters & Actions")
    st.write(f"Selected samples: **{len(sel_samples)}**")
    st.write(f"Min cluster size: **{min_cluster_size}**")
    st.markdown("---")
    st.markdown("### Legend")
    st.markdown(
        "<div style='display:flex; gap:8px; align-items:center'>"
        "<div style='width:14px;height:14px;background:#636EFA;border-radius:3px;'></div> Known &nbsp;"
        "<div style='width:14px;height:14px;background:#FF6363;border-radius:3px;'></div> Novel</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    if st.button("Export Top-20 FASTA"):
        outbuf = io.StringIO()
        if reps:
            for r in reps[:20]:
                outbuf.write(f">{r.id}\n{str(r.seq)}\n")
            st.download_button("Download top-20 FASTA", data=outbuf.getvalue(), file_name="top20_reps.fa")
        else:
            st.info("No representative FASTA available.")

# CENTER: UMAP/PCA + taxonomy + novel candidates
with main_col:
    st.subheader("UMAP / PCA View")
    # build umap_df (use columns if present, else PCA on embeddings, else random)
    if {'umap1', 'umap2'}.issubset(set(c.lower() for c in clusters_df.columns)):
        c1 = [c for c in clusters_df.columns if c.lower() == "umap1"][0]
        c2 = [c for c in clusters_df.columns if c.lower() == "umap2"][0]
        umap_df = clusters_df.rename(columns={c1: "UMAP1", c2: "UMAP2"}).reset_index(drop=True)
    elif emb is not None:
        emb_small = emb if emb.shape[1] <= 256 else emb[:, :256]
        coords = PCA(n_components=2, random_state=42).fit_transform(emb_small)
        umap_df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
        if 'cluster' in clusters_df.columns and len(umap_df) == len(clusters_df):
            umap_df['cluster'] = clusters_df['cluster'].values
    else:
        n = max(asvs_found, 200)
        rng = np.random.default_rng(42)
        umap_df = pd.DataFrame({"UMAP1": rng.normal(size=n), "UMAP2": rng.normal(size=n), "cluster": np.random.randint(0, 10, size=n)})

    # key column heuristics
    if 'rep' in clusters_df.columns:
        umap_df['key'] = clusters_df['rep'].astype(str).values[:len(umap_df)]
    elif 'index' in clusters_df.columns:
        umap_df['key'] = clusters_df['index'].astype(str).values[:len(umap_df)]
    elif 'qseqid' in clusters_df.columns:
        umap_df['key'] = clusters_df['qseqid'].astype(str).values[:len(umap_df)]
    elif 'cluster' in clusters_df.columns:
        umap_df['key'] = clusters_df['cluster'].astype(str).values[:len(umap_df)]
    else:
        umap_df['key'] = umap_df.index.astype(str)

    # collect novel keys from blast_novel
    novel_keys = set()
    qid_col = None
    for cand in ("rep_key", "qseqid", "qseq_id", "q_id", "query"):
        if cand in blast_novel.columns:
            qid_col = cand
            break
    if qid_col and not blast_novel.empty:
        novel_keys.update(blast_novel[qid_col].astype(str).tolist())
    if 'classification' in blast_novel.columns and not blast_novel.empty:
        mask = blast_novel['classification'].astype(str).str.contains('novel', case=False, na=False)
        if qid_col:
            novel_keys.update(blast_novel.loc[mask, qid_col].astype(str).tolist())
        elif 'qseqid' in blast_novel.columns:
            novel_keys.update(blast_novel.loc[mask, 'qseqid'].astype(str).tolist())

    def normalize_key(k: str) -> str:
        k = str(k).strip()
        m = re.match(r"(cluster\d+)", k)
        if m:
            return m.group(1)
        if k.isdigit():
            return f"cluster{k}"
        m2 = re.search(r"(cluster\d+)", k)
        if m2:
            return m2.group(1)
        return k

    umap_df['key_norm'] = umap_df['key'].astype(str).apply(normalize_key)
    novel_keys_norm = set(normalize_key(x) for x in novel_keys)
    umap_df['is_novel'] = umap_df['key'].astype(str).isin(novel_keys) | umap_df['key_norm'].isin(novel_keys_norm)

    if len(novel_keys) == 0 and 'cluster' in umap_df.columns:
        try:
            umap_df['is_novel'] = umap_df['cluster'].apply(lambda x: int(x) == -1)
        except Exception:
            pass

    # apply filters
    display_df = umap_df.copy()
    if sel_samples and 'sample' in clusters_df.columns:
        idx = clusters_df[clusters_df['sample'].isin(sel_samples)].index
        display_df = display_df.loc[display_df.index.isin(idx)]
    if min_cluster_size and 'cluster' in clusters_df.columns and not summary_df.empty:
        small = summary_df[summary_df['size'] < min_cluster_size]['cluster'].astype(str).tolist()
        display_df = display_df[~display_df['cluster'].astype(str).isin(small)]
    if novelty_filter == "Known only":
        display_df = display_df[display_df['is_novel'] == False]
    elif novelty_filter == "Novel only":
        display_df = display_df[display_df['is_novel'] == True]

    display_df['status'] = display_df['is_novel'].map({True: "Novel", False: "Known"})
    display_df['draw_order'] = display_df['status'].map({"Known": 0, "Novel": 1})
    display_df = display_df.sort_values("draw_order")

    fig = px.scatter(
        display_df,
        x="UMAP1",
        y="UMAP2",
        color="status",
        color_discrete_map={"Known": "#636EFA", "Novel": "#FF6363"},
        category_orders={"status": ["Known", "Novel"]},
        height=520,
        width=900,
        labels={"color": "Status"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.9))
    st.plotly_chart(fig, use_container_width=True)

    # taxonomy & novel candidates panel
    st.markdown("### Taxonomic composition")
    if not blast_annot.empty:
        tax_col = None
        for c in blast_annot.columns:
            if any(k in c.lower() for k in ["species", "sname", "ssciname", "stitle", "tax", "desc"]):
                tax_col = c
                break
        if tax_col:
            agg = blast_annot[tax_col].value_counts().head(10)
            pie = px.pie(values=agg.values, names=agg.index, title=f"Top taxa ({tax_col})")
            st.plotly_chart(pie, use_container_width=True)
        elif "sseqid" in blast_annot.columns:
            agg = blast_annot["sseqid"].value_counts().head(10)
            pie = px.pie(values=agg.values, names=agg.index, title="Top BLAST hits (sseqid)")
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No taxonomic or subject ID column found in BLAST file.")
    else:
        st.info("No BLAST annotated file found.")

    st.markdown("### Top novel candidates")
    if not blast_novel.empty:
        st.dataframe(blast_novel.head(50))
    else:
        st.info("No novel BLAST entries present.")

# RIGHT: inspector, rep display, chat injection
with right_col:
    st.subheader("Inspector")
    cluster_choices = []
    if not summary_df.empty and 'cluster' in summary_df.columns:
        cluster_choices = sorted(summary_df['cluster'].astype(str).tolist())
    elif 'cluster' in clusters_df.columns:
        cluster_choices = sorted(clusters_df['cluster'].astype(str).unique().tolist())
    sel_cluster = st.selectbox("Select cluster", options=cluster_choices if cluster_choices else ["None"])
    if sel_cluster and sel_cluster != "None":
        st.write(f"**Cluster:** {sel_cluster}")
        if not summary_df.empty and 'cluster' in summary_df.columns:
            row = summary_df[summary_df['cluster'].astype(str) == sel_cluster]
            if not row.empty:
                for col in row.columns:
                    st.write(f"**{col}**: {row.iloc[0][col]}")

        # representative sequence
        rep_found = None
        for r in reps:
            hid = r.id
            if hid.startswith("cluster") and hid.split("|")[0].replace("cluster", "") == sel_cluster:
                rep_found = r
                break
            if hid == sel_cluster or hid.split("|")[0] == sel_cluster:
                rep_found = r
                break

        if rep_found:
            st.markdown("**Representative sequence**")
            st.code(f">{rep_found.id}\n{str(rep_found.seq)}")
            st.download_button("Download rep FASTA", data=f">{rep_found.id}\n{str(rep_found.seq)}\n", file_name=f"cluster_{sel_cluster}_rep.fa")
        else:
            st.info("Representative sequence not found in FASTA.")

        # BLAST hits
        if not blast_annot.empty and 'qseqid' in blast_annot.columns:
            hits = blast_annot[blast_annot['qseqid'].astype(str).str.contains(str(sel_cluster))]
            if not hits.empty:
                st.markdown("**BLAST hits (top)**")
                st.dataframe(hits.head(10))

        # curator note + validation
        st.markdown("---")
        note = st.text_input("Curator note", key="note_input")
        if st.button("Mark as validated"):
            save_validation(sel_cluster, note=note)
            st.success(f"Marked cluster {sel_cluster} as validated.")

    # Chat injection (enable_chat controls overall rendering)
    if enable_chat:
        # Build floating and inline iframe snippets
        floating_iframe = f'''
        <iframe
          src="{BP_SHARE_URL}"
          style="
            position: fixed;
            right: 18px;
            bottom: 18px;
            width: 360px;
            height: 620px;
            border: none;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            z-index: 999999;
            overflow: hidden;
          "
          allow="microphone; camera; autoplay; clipboard-read; clipboard-write"
          sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals allow-storage-access-by-user-activation"
        ></iframe>
        '''

        inline_iframe = f'''
        <iframe
          src="{BP_SHARE_URL}"
          style="
            width:100%;
            height:680px;
            border:none;
            border-radius:8px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.10);
            overflow:hidden;
          "
          allow="microphone; camera; autoplay; clipboard-read; clipboard-write"
          sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals allow-storage-access-by-user-activation"
        ></iframe>
        '''

        try:
            if render_chat_inline:
                # Inline rendering in right column
                if inline_lazy_load:
                    if st.button("Open Assistant Chat (inline)"):
                        components.html(inline_iframe, height=680, scrolling=True)
                else:
                    components.html(inline_iframe, height=680, scrolling=True)
            else:
                # Floating render (keeps the floating bubble & reserves a small height)
                # Reserve small height so layout is stable (actual iframe is fixed-positioned)
                components.html(floating_iframe, height=1, scrolling=False)
        except Exception as e:
            st.warning(f"Chat widget could not be injected: {e}")

# Bottom: detailed summary and exports
st.markdown("---")
st.header("Detailed outputs & exports")
colA, colB = st.columns([3, 1])
with colA:
    st.subheader("Cluster summary")
    if not summary_df.empty:
        st.dataframe(summary_df.head(200))
    else:
        st.info("No cluster summary file found.")
with colB:
    st.subheader("Exports")
    if st.button("Export results ZIP"):
        zipbuf = io.BytesIO()
        with ZipFile(zipbuf, "w") as z:
            for p in [CLUSTERS_TSV, CLUSTER_SUM, REPS_FASTA, BLAST_ANNOT, BLAST_NOVEL]:
                if p.exists():
                    z.write(p, arcname=os.path.basename(p))
        zipbuf.seek(0)
        st.download_button("Download results ZIP", data=zipbuf, file_name="deepsea_results.zip")

st.caption("Pipeline metadata: container tag: biobloom:v1.5.2 • Run timestamp: autogenerated")
