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
            #G0 = np.random.normal(0, 1, (a.shape[0], b.shape[0]))
            #G0 = np.eye(a.shape[0])
            # G0 = np.zeros((a.shape[0], b.shape[0]))
            # # 随机选择num_ones个位置，并将其设为1
            # ones_indices = np.random.choice(a.shape[0]*b.shape[0], 2*a.shape[0], replace=False)
            # G0.flat[ones_indices] = 1
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