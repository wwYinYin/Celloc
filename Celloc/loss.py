import torch

def exp_loss(M, G):
    return torch.sum(M * G)

def space_loss2(constC, hC1, hC2, G):
    tens = tensor_product(constC, hC1, hC2, G)
    return torch.sum(tens * G)

def tensor_product(constC, hC1, hC2, G):
    A = - torch.matmul(
        torch.matmul(hC1, G), hC2.T
    )
    tens = constC + A
    # tens -= tens.min()
    return tens

def space_loss(D_A, D_B, G):
    n = D_A.shape[0]
    m = D_B.shape[0]
    results=0
    for i in range(n):
        for k in range(n):
            for j in range(m):
                for l in range(m):
                    space_diff = (D_A[i, k] - D_B[j, l]) ** 2
                    results += space_diff * G[i, j] * G[k, l]
    return results

def space_loss_optimized(space_diff, G):
    n = space_diff.shape[0]
    m = space_diff.shape[1]
    G = G.view(n, m, 1, 1)
    results = torch.sum(space_diff * G * G.permute(0, 2, 1, 3))

    return results

def rotation_loss(new_G_probs,coor_A,coor_B):
    H = torch.mm(coor_B.T, torch.mm(new_G_probs.T, coor_A))  # (2,2)
    U, S, Vt = torch.svd(H)
    R = torch.mm(Vt.T, U.T)
    new_coor_B = torch.mm(R, coor_B.T).T
    # dist_result=0
    # for i in range(new_G_probs.shape[0]):
    #     for j in range(new_G_probs.shape[1]):
    #         dist_result += new_G_probs[i,j]*torch.norm(coor_A[i,:] - new_coor_B[j,:])
    # return dist_result
    coor_A_diff = coor_A[:, None, :] - new_coor_B[None, :, :]
    dist_result = (new_G_probs * torch.norm(coor_A_diff, dim=2)).sum()
    return dist_result