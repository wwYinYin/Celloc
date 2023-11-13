import numpy as np
from typing import Optional, Tuple
from anndata import AnnData
from .utils import euclidean_distances, extract_data_matrix, to_dense_array, kl_divergence_backend, \
                    pcc_distances
from . import mapping_optimizer as mo
from typing import List, Tuple, Optional


def map_cell_to_space(
    M,
    D_A,
    D_B,
    a,
    b,
    use_gpu = True,
    learning_rate=0.001,
    num_epochs=500,
    alpha=0.01,
    lambda_rate=0.1,
    verbose=True,
    task="mapping"
):
    if verbose:
        print_each = 10
    else:
        print_each = None

    mapper = mo.MapperConstrained(M=M, D_A=D_A, D_B=D_B, a=a,b=b,
            use_gpu=use_gpu, alpha=alpha, lambda_rate=lambda_rate,task=task)
    mapping_matrix = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )
    
    return mapping_matrix


def paired_align(
    sc_adata: AnnData, 
    spatial_adata: AnnData, 
    sc_dissimilarity: str = 'pcc',
    sp_dissimilarity: str = 'kl', 
    use_rep: bool = True,
    b_init=None,
    use_gpu: bool = True,
    task="mapping") -> Tuple[np.ndarray, Optional[int]]:
    
    # Calculate spatial distances
    if use_rep:
        sc_exp=sc_adata.obsm['X_pca']
    else:
        sc_exp=sc_adata.X
    coordinates = spatial_adata.obsm['spatial'].copy()

    if sc_dissimilarity.lower() == 'euclidean':
        D_sc = euclidean_distances(sc_exp, sc_exp)
    elif sc_dissimilarity.lower() == 'pcc':
        D_sc = pcc_distances(sc_exp, sc_exp)
    elif sc_dissimilarity.lower() == 'kl':
        sc_exp = sc_exp + 0.01
        D_sc = kl_divergence_backend(sc_exp, sc_exp)
    D_sp = euclidean_distances(coordinates, coordinates)
    D_sc=np.abs(D_sc)
    D_sc=(D_sc/D_sc.max())*10
    D_sp=(D_sp/D_sp.max())*10
    
    # Calculate expression dissimilarity
    sc_X = to_dense_array(extract_data_matrix(sc_adata, rep=None))
    sp_X = to_dense_array(extract_data_matrix(spatial_adata, rep=None))
    if sp_dissimilarity.lower() == 'euclidean' or sp_dissimilarity.lower() == 'euc':
        M = euclidean_distances(sc_X, sp_X)
    elif sp_dissimilarity.lower() == 'kl':
        s_A = sc_X + 0.01
        s_B = sp_X + 0.01
        M = kl_divergence_backend(s_A, s_B)
    elif sp_dissimilarity.lower() == 'pcc':
        M = pcc_distances(sc_X, sp_X)

    # init distributions
    a=np.ones((sc_X.shape[0],))

    if b_init is not None:
        b=np.array(b_init)/sc_X.shape[0]
    else:
        b=np.ones((sp_X.shape[0],))/sc_X.shape[0]

    # Run
    mapping_matrix = map_cell_to_space(M, D_sc, D_sp, a, b, use_gpu = use_gpu,task=task)
    return mapping_matrix

