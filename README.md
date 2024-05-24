# Celloc
Accurate mapping between single-cell RNA sequencing (scRNA-seq) and low-resolution spatial transcriptomics (ST) data compensates for both the limited spatial resolution of ST spots and the inability of scRNA-seq to preserve spatial information. Here, we developed Celloc, deep learning non-convex optimization-based method for flexible single-cell-to-spot mapping, which enables either dissecting cell composition of each spot (regular mapping) or predicting spatial location of every cell in scRNA-seq data (greedy mapping). We benchmarked Celloc on simulated ST data where Celloc outperformed state-of-the-art methods in accuracy and robustness. Evaluations on real datasets suggested that Celloc could reconstruct the spatial pattern of cells in breast cancer, reveal spatial subclonal heterogeneity of ductal carcinoma in situ, infer spatial tumor-immune microenvironment, and signify spatial expression patterns in myocardial infarction. Together, the results suggest that Celloc can accurately reconstruct cellular spatial structures with various cell types across different histological regions.
## Install Guidelines
* We recommend you to use an Anaconda virtual environment. Install PyTorch >= 1.13 according to your GPU driver and Python >= 3.9, and run：

```
pip install -r requirements.txt
```
## Datasets
All datasets used in our paper can be found in:
* The simulated mouse cerebellum and hippocampus can be downloaded from [Cerebellum](https://drive.google.com/file/d/1qfz2T8u3HRG4qdZc9qafcO4aCvjA91Rb/view?usp=share_link) and [Hippocampus](https://drive.google.com/file/d/1Jyd14n-ISc5lF65pnJWLhCCgSkpjtbsr/view?usp=share_link).
* The HER2+ breast cancer scRNA-seq data is downloaded from [HER2+ breast cancer](https://drive.google.com/file/d/1G8gK4MxCmRG4JZi588wloMsP8iZlQf_z/view?usp=share_link).
* Two 10X Visium ST samples (ST sample1, ST sample2) are obtained from [ST sample1](https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0) and [ST sample2](https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard).
* The DCIS datasets are obtained from [CellTrek](https://github.com/navinlabcode/CellTrek).
* The scRNA-seq data of myocardial infarction is downloaded from [GSE129175](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129175).
* The ST data of myocardial infarction is downloaded from [GSE165857](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165857).

The way to read all this data has been given in 'Regular_Mapping_SC2ST.ipynb' and 'Greedy_Mapping_SC2ST.ipynb'.

## Tutorial
Celloc can (1) fill ST spots with suitable number of cells from scRNA-seq to enhance the quality of ST data in terms of resolution and gene expression quantity (regular mapping); (2) assign the spatial location for every cell in scRNA-seq data to completely investigate the spatial pattern across the full scRNA-seq dataset (greedy mapping).
### Input files
* A scRNA-seq data with annotated with cell types (.h5ad, .txt, .csv)
* A ST data with spatial coordinates (.h5ad, .txt, .csv)

### Regular single-cell-to-spot mapping
  ```
  Regular_mapping_tutorial/
    --Run_simulated_mouse_cerebellum.ipynb
    --Run_simulated_mouse_hippocampus.ipynb
    --Run_HER2+_with_STsample1.ipynb
    --Run_DCIS1_with_STsample1.ipynb
    --Run_HER2+_with_STsample2.ipynb
    --Run_DCIS1_with_STsample2.ipynb
    --Run_DCIS2.ipynb
    --Run_DCIS1.ipynb
    --Run_MI.ipynb
  ```
### Greedy single-cell-to-spot mapping
  ```
  Greedy_mapping_tutorial/
    --Run_simulated_mouse_cerebellum.ipynb
    --Run_simulated_mouse_hippocampus.ipynb
    --Run_MI.ipynb
  ```

## Results visualization
The visualization code for simulated mouse cerebellum and hippocampus in Supplementary Figures 3 and 4 has been given at the end of 'Regular_Mapping_SC2ST.ipynb'.  
For other 10X Visium ST data mapping results visualization tutorials, refer to the following code:  
  ```
  10X_Visium_Visulization.ipynb
  ```
