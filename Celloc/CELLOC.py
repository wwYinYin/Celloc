import numpy as np
from typing import Optional, Tuple
from anndata import AnnData
from .utils import euclidean_distances, extract_data_matrix, to_dense_array, kl_divergence_backend, \
                    pcc_distances
from . import mapping_optimizer as mo
from typing import List, Tuple, Optional
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import pandas as pd
from typing import Optional, Union, List
import scipy.sparse as sp
from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn.functional as F
import torch

def map_cell_to_space(
    M,
    D_A,
    D_B,
    a,
    b,
    use_gpu = True,
    learning_rate=0.001,
    num_epochs=500,
    alpha=0,
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
    new_sc_adata=embedding_feature(sc_adata)
    new_spatial_adata=embedding_feature(spatial_adata, k_cutoff=4)
    sc_rep=new_sc_adata.layers['GATE_ReX']
    sp_rep=new_spatial_adata.layers['GATE_ReX']

    if sp_dissimilarity.lower() == 'euclidean' or sp_dissimilarity.lower() == 'euc':
        M = euclidean_distances(sc_X, sp_X)
    elif sp_dissimilarity.lower() == 'kl':
        s_A = sc_X + 0.01
        s_B = sp_X + 0.01
        M = kl_divergence_backend(s_A, s_B)
    elif sp_dissimilarity.lower() == 'pcc':
        M = pcc_distances(sc_X, sp_X)

    M_ano = pcc_distances(sc_rep, sp_rep)
    M=M_ano+M
    # init distributions
    a=np.ones((sc_X.shape[0],))

    if b_init is not None:
        b=np.array(b_init)/sc_X.shape[0]
    else:
        b=np.ones((sp_X.shape[0],))/sc_X.shape[0]

    # Run
    mapping_matrix = map_cell_to_space(M, D_sc, D_sp, a, b, use_gpu = use_gpu,task=task)
    return mapping_matrix

def embedding_feature(adata,k_cutoff=6,self_loop=True):
    #Normalization
    device = torch.device('cuda')
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    adata = Cal_Spatial_Net(adata,  k_cutoff=k_cutoff)
    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    data = Transfer_pytorch_Data(adata_Vars)
    model = mo.GATE(hidden_dims = [data.x.shape[1]] + [512, 64]).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    n_epochs=500
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    
    model.eval()
    z, out = model(data.x, data.edge_index)
    
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata_Vars.obsm['embedding'] = STAGATE_rep

    ReX = out.to('cpu').detach().numpy()
    ReX[ReX<0] = 0
    adata_Vars.layers['GATE_ReX'] = ReX
    return adata_Vars

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data



def Cal_Spatial_Net(adata:AnnData,
                    k_cutoff:Optional[Union[None,int]]=None, 
                    return_data:Optional[bool]=True,
                    verbose:Optional[bool]=True
    ) -> AnnData:
    # 使用knn_graph函数计算K最近邻图
    if 'spatial' not in adata.obsm.keys():
        edge_index = knn_graph(x=torch.tensor(adata.obsm['X_pca'].copy()), flow='target_to_source',
                            k=k_cutoff, loop=True, num_workers=8)
    else:
        edge_index = knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                k=k_cutoff, loop=True, num_workers=8)
    # 确保图是无向的
    edge_index = to_undirected(edge_index, num_nodes=adata.shape[0]) 

    # 将edge_index转换为DataFrame
    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
    # 创建一个映射，将索引映射到观察名
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    # 使用映射更新Cell1和Cell2的值
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    # 将结果保存在adata.uns['Spatial_Net']中
    adata.uns['Spatial_Net'] = graph_df
    
    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0]/adata.n_obs} neighbors per cell on average.')

    if return_data:
        return adata