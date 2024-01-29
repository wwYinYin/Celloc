import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import calculate_squared_differences,init_matrix
from .loss import exp_loss, space_loss, rotation_loss,space_loss2,space_loss_optimized

class MapperConstrained:
    def __init__(
        self,
        M,
        D_A,
        D_B,
        a,
        b,
        G_init = None,
        use_gpu = True,
        alpha=0.1,
        lambda_rate=1,
        lambda_rotation=1,
        random_state=2024,
        task="mapping"
    ):
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(device)
        self.M = torch.tensor(M, device=device, dtype=torch.float32)
        self.a = torch.tensor(a, device=device, dtype=torch.float32)
        self.b = torch.tensor(b, device=device, dtype=torch.float32)

        self.alpha = alpha
        self.lambda_rate = lambda_rate
        self.lambda_rotation = lambda_rotation
        self.random_state = random_state
        self.task = task

        constC, hC1, hC2 = init_matrix(D_A, D_B, a, b)
        self.constC = torch.tensor(constC, device=device, dtype=torch.float32)
        self.hC1 = torch.tensor(hC1, device=device, dtype=torch.float32)
        self.hC2 = torch.tensor(hC2, device=device, dtype=torch.float32)

        if G_init is None:
            G0 = a[:, None] * b[None, :] #得到[m×n]的初始化矩阵
        else:
            G0 = (1/np.sum(G_init)) * G_init
        self.G = torch.tensor(
            G0, device=device, requires_grad=True, dtype=torch.float32
        )
        # self.space_diff = calculate_squared_differences(D_A,D_B) #[n,n,m,m]
        # self.space_diff = torch.tensor(self.space_diff, device=device, dtype=torch.float32)
        self.D_A = torch.tensor(D_A, device=device, dtype=torch.float32)
        self.D_B = torch.tensor(D_B, device=device, dtype=torch.float32)
        self.d_loss = nn.L1Loss()

    def _loss_fn(self, verbose=True):
        G_probs = F.relu(self.G)
        if self.task == 'mapping':
            G_probs = F.softmax(G_probs, dim=1)
        elif self.task == 'location_recovery':
            G_probs = (G_probs.T/G_probs.sum(axis=1)).T
            G_probs=torch.where(torch.isnan(G_probs), torch.full_like(G_probs, 0), G_probs)
        
        # cell_d_pred = torch.sum(G_probs, dim=1)
        # cell_density_term = self.lambda_rate * self.d_loss(cell_d_pred, self.a*len(self.a))
        d_pred = torch.sum(G_probs, dim=0)
        density_term = self.lambda_rate * self.d_loss(d_pred, self.b*len(self.a))
        # d_pred = torch.sum(G_probs, dim=1)
        # density_term = self.lambda_rate * self.d_loss(d_pred, self.a)

        expression_term = (1-self.alpha) * exp_loss(self.M, G_probs/len(self.a))
        #space_term = self.alpha * space_loss(self.D_A, self.D_B, G_probs)
        #space_term = self.alpha * space_loss_optimized(self.space_diff, G_probs/len(self.a))
        space_term = self.alpha * space_loss2(self.constC, self.hC1, self.hC2, G_probs/len(self.a))

        #space2_term = self.lambda_rotation*rotation_loss(G_probs/len(self.a), self.coor_A, self.coor_B)


        expression_lossvalue = expression_term.tolist()
        space_lossvalue = space_term.tolist()
        #cell_density_loss = cell_density_term.tolist()
        density_loss = density_term.tolist()
        #density_loss=0
        if verbose:
            term_numbers = [expression_lossvalue, space_lossvalue, density_loss]
            term_names = ["expression_term", "space_term", "density_term"]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        total_loss = expression_term + space_term + density_term
        #total_loss = expression_term + space_term

        return (
            total_loss,
            G_probs,
        )

    def train(self, num_epochs, learning_rate=0.001, print_each=100):
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.G], lr=learning_rate,weight_decay=1e-6)
        #optimizer = torch.optim.Adam([self.G], lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)
            loss = run_loss[0]
            result=run_loss[-1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
        # take final softmax w/o computing gradients
        with torch.no_grad():
            # G_probs = F.relu(self.G)
            # output = F.softmax(G_probs, dim=1).cpu().numpy()
            output = result.cpu().numpy()
            return output


from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Union, Tuple, Optional,List,Any

class GATE(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)
    
class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)


        self._alpha = None
        self.attentions = None


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        #alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)