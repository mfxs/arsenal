# Some important functions and classes for neural networks
import math
import torch
import cvxpy as cp
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter
from sklearn.gaussian_process.kernels import RBF


# Adjacency matrix
def adjacency_matrix(X, mode, graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1, gpu=0):
    # RBF kernel function
    if mode == 'rbf':
        kernel = RBF(length_scale=scale)
        A = kernel(X, X)

    # Pearson correlation coefficient
    elif mode == 'pearson':
        A = np.corrcoef(X.T)

    # Sparse coding
    elif mode == 'sc':
        A = cp.Variable((X.shape[1], X.shape[1]))
        term1 = cp.norm(X * A - X, p='fro')
        term2 = cp.norm1(A)
        constraints = []
        for i in range(X.shape[1]):
            constraints.append(A[i, i] == 0)
            for j in range(X.shape[1]):
                constraints.append(A[i, j] >= 0)
        constraints.append(A == A.T)
        objective = cp.Minimize(term1 + graph_reg * term2)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        A = A.value
        A = A + self_con * np.eye(X.shape[1])

    # Omit small values
    A[np.abs(A) < epsilon] = 0

    # Normalization
    D = np.diag(np.sum(A, axis=1) ** (-0.5))
    A = np.matmul(np.matmul(D, A), D)
    A = torch.tensor(A, dtype=torch.float32, device=gpu)

    return A


# MyDataset
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label):
        self.data, self.label = data, label

    # Get item
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    # Get length
    def __len__(self):
        return self.data.shape[0]


# Graph convolution
class GraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y):
        super(GraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.weight = Parameter(torch.FloatTensor(dim_X, dim_y))
        self.reset_parameters()

    # Forward propagation
    def forward(self, X, adj):
        res = torch.matmul(adj, torch.matmul(X, self.weight))
        return res

    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
