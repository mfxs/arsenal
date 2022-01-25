"""
This is a collection of many useful machine learning models
Thus, it is named as 'Arsenal'!

The content of this collection is shown as below:

=====Regression=====
OLS: Ordinary Least Square
RR: Ridge Regression
LASSO: Least Absolute Shrinkage and Selection Operator
PLSR: Partial Least Square Regression
GPR: Gaussian Process Regression
ELM: Extreme Learning Machine
MC-GCN: Multi-Channel Graph Convolutional Networks
GC-LSTM: Graph Convolution Long Short-Term Memory

=====Classification=====
LR: Logistic Regression

=====Regression & Classification=====
FCN: Fully Connected Networks
LSTM: Long Short-Term Memory
GCN: Graph Convolutional Networks

=====Dimensionality Reduction=====
PCA: Principal Component Analysis
t-SNE: t-distributed Stochastic Neighbor Embedding
AE: Auto-Encoders
VAE: Variational Auto-Encoders

To be continued ...
"""

# Load packages
import time
import math
import torch
import warnings
import cvxpy as cp
import numpy as np
import seaborn as sns
from torch import nn, optim
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error

# Ignore warnings
warnings.filterwarnings('ignore')


# Load dataset
def load_data(data_type='regression', test_size=0.3, seed=123, normalization=True):
    # Regression dataset or classification dataset
    if data_type == 'regression':
        X, y = load_diabetes(return_X_y=True)
    elif data_type == 'classification':
        X, y = load_digits(return_X_y=True)
    else:
        raise Exception('You have given a wrong data type.')

    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=test_size, random_state=seed)

    # Whether to normalize the dataset
    if normalization:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Plot prediction and ground truth
def plot_pred(y_pred, y_test, title='Title', figsize=(10, 10), dpi=150):
    # Compute the performance index
    r2 = 100 * r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Plot the prediction scatter
    plt.figure(figsize=figsize, dpi=dpi)
    plt.suptitle(title + ' (R2: {:.2f}%, RMSE: {:.3f})'.format(r2, rmse))
    plt.subplot(211)
    plt.scatter(y_test, y_pred, label='Samples')
    plt.plot(y_test, y_test, 'r', label='Isoline')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.grid()
    plt.legend()

    # Plot the prediction curve
    plt.subplot(212)
    plt.plot(y_test, label='Ground Truth')
    plt.plot(y_pred, label='Prediction')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.show()


# Plot confusion matrix
def confusion(y_pred, y_test, title='Title'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, cmap='YlGnBu', annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title(title)
    plt.show()


# Plot scatter
def scatter(pc, color, title='Title'):
    plt.figure()
    plt.scatter(pc[:, 0], pc[:, 1], c=color, cmap='tab10')
    plt.xlabel('Component_1')
    plt.ylabel('Component_2')
    plt.title(title)
    plt.show()


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

    else:
        raise Exception('Wrong mode selection.')

    # Omit small values
    A[np.abs(A) < epsilon] = 0

    # Normalization
    D = np.diag(np.sum(A, axis=1) ** (-0.5))
    A = np.matmul(np.matmul(D, A), D)
    A = torch.tensor(A, dtype=torch.float32).cuda(gpu)

    return A


# MyDataset
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label, gpu=0):
        super(MyDataset, self).__init__()
        self.gpu = gpu
        self.data = self.__transform__(data)
        self.label = self.__transform__(label)

    # Transform
    def __transform__(self, data):
        return torch.tensor(data, dtype=torch.float32).cuda(self.gpu)

    # Get item
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    # Get length
    def __len__(self):
        return self.data.shape[0]


# Graph convolution
class GraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj):
        super(GraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.weight = Parameter(torch.FloatTensor(dim_X, dim_y))
        self.reset_parameters()

    # Forward propagation
    def forward(self, X):
        res = torch.matmul(self.adj, torch.matmul(X, self.weight))
        return res

    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)


class NeuralNetwork(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    # def training(self):
