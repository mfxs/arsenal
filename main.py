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
import warnings
import cvxpy as cp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
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
