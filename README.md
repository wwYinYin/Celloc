# Celloc
Accurate mapping between single-cell RNA sequencing (scRNA-seq) and low-resolution spatial transcriptomics (ST) data compensates for both the limited spatial resolution of ST spots and the inability of scRNA-seq to preserve spatial information. Here, we developed Celloc, deep learning non-convex optimization-based method for flexible single-cell-to-spot mapping, which enables either dissecting cell composition of each spot (regular mapping) or predicting spatial location of every cell in scRNA-seq data (greedy mapping). We benchmarked Celloc on simulated ST data where Celloc outperformed state-of-the-art methods in accuracy and robustness. Evaluations on real datasets suggested that Celloc could reconstruct the spatial pattern of cells in breast cancer, reveal spatial subclonal heterogeneity of ductal carcinoma in situ, infer spatial tumor-immune microenvironment, and signify spatial expression patterns in myocardial infarction. Together, the results suggest that Celloc can accurately reconstruct cellular spatial structures with various cell types across different histological regions.
## Install Guidelines
* We recommend you to use an Anaconda virtual environment. Install PyTorch >= 1.13 according to your GPU driver and Python >= 3.9, and run：

```
pip install -r requirements.txt
```

