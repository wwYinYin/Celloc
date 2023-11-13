import numpy as np
import pandas as pd
import scanpy as sc

import datatable as dt

import os
from pathlib import Path
import random
import anndata as ad
import warnings

def read_file(file_path):
    # Read file
    try:
        file_delim = "," if file_path.endswith(".csv") else "\t"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            file_data = dt.fread(file_path, header=True)
            colnames = pd.read_csv(file_path, sep=file_delim, nrows=1, index_col=0).columns
            rownames = file_data[:, 0].to_pandas().values.flatten()
            file_data = file_data[:, 1:].to_pandas()
            file_data.index = rownames
            file_data.columns = colnames
   
    except Exception as e:
        raise IOError("Make sure you provided the correct path to input files. "
                      "The following input file formats are supported: .csv with comma ',' as "
                      "delimiter, .txt or .tsv with tab '\\t' as delimiter.")

    return file_data


def read_data(scRNA_path, cell_type_path, cell_type_numbers_estimation_path, st_path, coordinates_path,celltype_col):
    if st_path.endswith("h5ad"):
        spatial_adata= sc.read_h5ad(st_path)
        coordinates_data=pd.DataFrame(spatial_adata.obsm['spatial'],index=spatial_adata.obs_names,columns=['X','Y'])
        spatial_adata.obs_names=['SPOT_'+str(col) for col in spatial_adata.obs_names]
        spatial_adata.var_names=['GENE_'+str(col) for col in spatial_adata.var_names]
    else:   
        st_data = read_file(st_path)
        st_data = st_data[~st_data.index.duplicated(keep=False)]
        coordinates_data = read_file(coordinates_path)
        
        st_data.columns = ['SPOT_'+str(col) for col in st_data.columns]
        st_data.index = ['GENE_'+str(idx) for idx in st_data.index]
        spatial_adata=ad.AnnData(st_data.T)
        spatial_adata.obsm['spatial']=np.array(coordinates_data)
    if scRNA_path.endswith("h5ad"):
        sc_adata = sc.read_h5ad(scRNA_path)
        sc_adata.obs_names=['CELL_'+str(col) for col in sc_adata.obs_names]
        sc_adata.var_names=['GENE_'+str(col) for col in sc_adata.var_names]
        sc_adata.obs['CellType']=['TYPE_'+str(cell) for cell in list(sc_adata.obs[celltype_col])]
    else:
        scRNA_data = read_file(scRNA_path)
        scRNA_data.columns = ['CELL_'+str(col) for col in scRNA_data.columns]
        scRNA_data.index = ['GENE_'+str(idx) for idx in scRNA_data.index]
        scRNA_data = scRNA_data[~scRNA_data.index.duplicated(keep=False)]

        cell_type_data = read_file(cell_type_path)
        cell_type_data.index = ['CELL_'+str(idx) for idx in cell_type_data.index]
        #cell_type_data.iloc[:,0] = ['TYPE_'+str(cell) for cell in list(cell_type_data.iloc[:,0])]
    
        sc_adata=ad.AnnData(scRNA_data.T)
        sc_adata.obs=cell_type_data
        sc_adata.obs['CellType']=['TYPE_'+str(cell) for cell in list(sc_adata.obs[celltype_col])]

    if cell_type_numbers_estimation_path is not None:
        cell_type_number_eachspot_data = read_file(cell_type_numbers_estimation_path)
        cell_type_number_eachspot_data.columns = ['TYPE_'+str(col) for col in cell_type_number_eachspot_data.columns]
        cell_type_number_eachspot_data.index = ['SPOT_'+str(idx) for idx in cell_type_number_eachspot_data.index]
    else:
        cell_type_number_eachspot_data = None

    return sc_adata, spatial_adata, cell_type_number_eachspot_data, coordinates_data

def sample_single_cells(sc_adata, cell_type_numbers_int, sampling_method, seed):   
    np.random.seed(seed)
    random.seed(seed)

    # Down/up sample of scRNA-seq data according to estimated cell type fractions
    # follow the order of cell types in cell_type_numbers_int
    unique_cell_type_labels = cell_type_numbers_int.index.values

    sampled_index_total = [] # List of 1D np.array of single cell indices
    for cell_type in unique_cell_type_labels:
        cell_type_index = list(np.where(sc_adata.obs['CellType'] == cell_type)[0])
        cell_type_count_available = len(cell_type_index)
        if cell_type_count_available == 0:
            raise ValueError(f"Cell type {cell_type} in the ST dataset is not available in the scRNA-seq dataset.")
        cell_type_count_desired = cell_type_numbers_int.loc[cell_type][0]
        if sampling_method == "duplicates":
            if cell_type_count_desired > cell_type_count_available:
                cell_type_selected_index = np.concatenate([
                    cell_type_index, np.random.choice(cell_type_index, cell_type_count_desired - cell_type_count_available)
                ], axis=0) # ensure at least one copy of each, then sample the rest

            else:
                cell_type_selected_index = random.sample(cell_type_index, cell_type_count_desired)
        
            sampled_index_total.append(cell_type_selected_index)
        
        else:
            raise ValueError("Invalid sampling_method provided")
    
    sampled_index_total = np.concatenate(sampled_index_total, axis=0).astype(int)
    all_cells_save_adata = sc_adata[sampled_index_total,:]

    return all_cells_save_adata
def normalize_data(data):
    data = np.nan_to_num(data).astype(float)
    data *= 10**6 / np.sum(data, axis=0, dtype=float)
    np.log2(data + 1, out=data)
    np.nan_to_num(data, copy=False)
    return data
    
#估计每个spot中的细胞数量
def estimate_cell_number_RNA_reads(st_data, mean_cell_numbers):
    # Read data
    expressions = st_data.values.astype(float)

    # Data normalization
    expressions_tpm_log = normalize_data(expressions)

    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)

    min_cell_numbers = 1 if min_RNA_reads > 0 else 0

    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)

    return cell_number_to_node_assignment


def  plot_results_bulk_ST_by_spot(assigned_locations, coordinates_data, dir_out, output_prefix="Predict", geometry='honeycomb', num_cols=3):
 
    fout_pdf_all = os.path.join(dir_out, f"{output_prefix}_cell_type_assignments_by_spot.pdf")
    metadata = assigned_locations.loc[:, ['Predict', 'CellType']].value_counts().unstack(fill_value=0)\
                    .reindex(index=assigned_locations.Predict.unique(), columns=assigned_locations.CellType.unique())
    metadata = metadata.astype(int)
    cell_types = list(metadata.columns)
    cell_types = list(np.sort(cell_types))
    #cell_types.insert(0,'Total cells')

    coordinates = coordinates_data.loc[metadata.index,:]
    X = coordinates.iloc[:,0]
    Y = coordinates.iloc[:,1]
    
    scale = Y.max() < 500 and ((Y - Y.round()).abs() < 1e-5).all()

    # define representative interval between each adjacent spot/point
    y_int = 1 if scale else np.median(np.unique(np.diff(np.sort(np.unique(Y)))))
    x_int = 1 if scale else np.median(np.unique(np.diff(np.sort(np.unique(X)))))
    print(scale)
    if geometry == 'honeycomb' and scale:
        print('Detecting row and column indexing of Visium data; rescaling for coordinates')
        
        #Rotate
        # X_prev = X
        # Y_prev = Y
        # X = Y_prev
        # Y = 1-X_prev
        
        # Rescale
        #Y = 1.75*Y

    elif geometry == 'square' and scale:
        print('Detecting row and column indexing of legacy ST data; rotating for coordinates')
        
        # Rotate
        # X_prev = X
        # Y_prev = Y
        # X = Y_prev
        # Y = 1-X_prev

    else:        
        # Rotate 
        Y = 1-Y

    if geometry == 'honeycomb':
        hex_vert = 6

        if scale:
            hex_rot = 0
            hex_rad = y_int
            hex_rad = 0.75*hex_rad

        else:
            hex_rot = 0
            hex_rad = x_int
            hex_rad = 1.2*hex_rad

    elif geometry == 'square':
        hex_vert = 4
        hex_rot = 45
        interval = y_int
        hex_rad = 1*np.sqrt(2*interval**2)

    else:
        print("Unknown geometry specified.")
        exit()

    num_rows = int(len(cell_types)/num_cols)
    if num_rows*num_cols < len(cell_types):
        num_rows = num_rows+1
    width = max(X)-min(X)
    height = max(Y)-min(Y)

    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size':'12'})
    plt.rcParams['figure.dpi'] = 450
    plt.rcParams['savefig.dpi'] = 450

    fig, axes = plt.subplots(num_rows,num_cols,figsize=(width/height*3*num_cols,3*num_rows))
    full_frac = 0.047 * (3*num_rows / (width/height*3*num_cols))
    k = 0
    for i in range(num_rows):
        for j in range(num_cols):

            ax = axes[i,j]

            if k >= len(cell_types):
                ax.axis('off')
            else:
                ct = cell_types[k]
                ax.set_aspect('equal')

                node_assignment = metadata.loc[:,ct]

                viridis = cm.get_cmap('viridis')
                #norm = matplotlib.colors.Normalize(min(node_assignment), max(node_assignment))
                norm = matplotlib.colors.Normalize(0, 7)
                node_assignment = (1/max(node_assignment))*node_assignment
                colors = viridis(node_assignment)

                for x, y, c in zip(X, Y, colors):
                    hex = RegularPolygon((x, y), numVertices=hex_vert, radius=hex_rad, 
                                         orientation=np.radians(hex_rot), 
                                         facecolor=c, edgecolor=None)
                    ax.add_patch(hex)

                # Also add scatter points in hexagon centres - not sure why this line has to be here
                ax.scatter(X, Y, c=[c[0] for c in colors],alpha=0)
                ct_label = ct
                ax.set_title(ct_label)
                cax = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=viridis),ax=ax,label='',fraction=0.036, pad = 0.04)
                ax.axis('off')

            k += 1

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle('Number of cells per spot mapped by CytoSPACE')        
    fig.savefig(fout_pdf_all, facecolor="w", bbox_inches='tight')