# 🌊 DeepSea eDNA AI Pipeline  

<<<<<<< HEAD
**AI-driven pipeline for deep-sea biodiversity discovery using environmental DNA (eDNA).**  
This project processes raw eDNA sequences, generates embeddings with **DNABERT**, performs **unsupervised clustering** to identify novel taxa, annotates sequences with **BLAST**, and provides an **interactive Streamlit dashboard** for exploration and visualization.  

---

## ✨ Features
- 📥 **Data ingestion & preprocessing** – filter, dereplicate, clean eDNA sequences.  
- 🧬 **Embeddings** – convert DNA sequences into numeric vectors with DNABERT (6-mer).  
- 🔍 **Clustering** – use UMAP + HDBSCAN to group sequences into taxa-like clusters.  
- 🧪 **Annotation** – BLAST consensus sequences against NCBI for known species mapping.  
- 🚀 **Novelty detection** – flag clusters with weak/no BLAST matches as potential new taxa.  
- 📊 **Visualization** – interactive UMAP plots, species composition pie charts, and top novel candidate tables.  
- 🖥️ **Streamlit dashboard** – explore results, inspect clusters, download FASTA, and export reports.  

---

## 🛠️ Tech Stack
- **Languages & Frameworks:** Python, Streamlit  
- **AI/ML:** PyTorch, HuggingFace Transformers (DNABERT), scikit-learn, UMAP, HDBSCAN  
- **Bioinformatics:** Biopython, MAFFT, NCBI BLAST+  
- **Data Processing:** NumPy, Pandas, SciPy, tqdm  
- **Visualization:** Plotly, Matplotlib  
- **Workflow & Exports:** Zipfile, OpenPyXL, Requests  

---

## 📂 Project Structure
```
deepsea_edna/
├── data/                # Input FASTA, embeddings, clusters, BLAST results
│   ├── preprocess/      # Preprocessed sequences
│   ├── DNABERT_embeddings/
│   ├── CLUSTER_files/
│   ├── BLAST_files/
├── results/             # Final reports, exports, UMAP plots
├── scripts/             # Core pipeline scripts
│   ├── filter_stream_prompt_nt.py   # Download + filter raw FASTA
│   ├── ref_preprocess_nt_uncultured.py
│   ├── 01_embed_dnabert6.py         # Generate embeddings
│   ├── 02_reduce_cluster.py         # Clustering + UMAP
│   ├── 03_consensus_blast.py        # Consensus + BLAST annotation
├── app.py               # Streamlit dashboard
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/deepsea_edna.git
cd deepsea_edna
```

### 2. Create a Conda environment
```bash
conda create -n edna_ai python=3.10 -y
conda activate edna_ai
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install external tools
You’ll need these installed and accessible on your system `PATH`:  
- **MAFFT** (for multiple sequence alignment)  
  ```bash
  brew install mafft        # macOS
  sudo apt-get install mafft  # Linux
  ```
- **NCBI BLAST+** (for sequence annotation)  
  ```bash
  brew install blast
  sudo apt-get install ncbi-blast+
  ```

---

## 🧪 Usage

### Step 1: Download & filter raw sequences
```bash
python scripts/filter_stream_prompt_nt.py
```

### Step 2: Preprocess sequences
```bash
python scripts/ref_preprocess_nt_uncultured.py
```

### Step 3: Generate DNABERT embeddings
```bash
python scripts/01_embed_dnabert6.py
```

### Step 4: Cluster sequences (UMAP + HDBSCAN)
```bash
python scripts/02_reduce_cluster.py
```

### Step 5: Build consensus & annotate with BLAST
```bash
python scripts/03_consensus_blast.py
```

### Step 6: Launch the dashboard
```bash
streamlit run app.py
```

Then open the link in your browser (`http://localhost:8501`).

---

## 📊 Output Files
- **Embeddings** → `data/DNABERT_embeddings/windows_embeddings.npy`  
- **Clusters** → `data/CLUSTER_files/clusters.tsv`, `cluster_summary.tsv`, `cluster_reps.fa`  
- **Consensus & BLAST results** → `data/BLAST_files/`  
- **Dashboard reports** → interactive on Streamlit, exportable as ZIP/FASTA  

---

## 🌍 Use Cases
- Deep-sea biodiversity monitoring (e.g., abyssal plains, hydrothermal vents).  
- Novel species discovery for **biotech/pharma bioprospecting**.  
- Environmental monitoring for **mining, fisheries, and climate projects**.  
- Conservation and compliance reporting for marine institutes.  

---

## 📈 Roadmap
- [ ] Integrate FAISS for scalable ANN search (millions of sequences).  
- [ ] Add ONNX export for lightweight embedding inference.  
- [ ] Expand dashboard with timeline comparisons across voyages.  
- [ ] Cloud-native pipeline (S3 + Prefect/Airflow orchestration).  

---

## 🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.  

---

## 📜 License
MIT License © 2025 Vivek Mishra
