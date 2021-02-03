# This is a collection of many useful machine learning models
# Thus, it is named 'Arsenal'
#
# The content of this collection is shown as below:
# 1--Ordinary Least Square (OLS)                                                               regression
# 2--Ridge Regression (RR)                                                                         regression
# 3--Least Absolute Shrinkage and Selection Operator (LASSO)           regression
# 4--Partial Least Square (PLS)                                                                   regression
# 5--Logistic Regression (LR)                                                                      classification
# 6--Principal Component Analysis (PCA)                                                 dimensionality reduction
# 7--t-distributed Stochastic Neighbor Embedding (t-SNE)                   dimensionality reduction
# 8--Fully Connected Networks (FCN)                                                       regression/classification
# 9--Extreme Learning Machine (ELM)                                                      regression
# 10--Long Short-Term Memory (LSTM)                                                  regression/classification
# 11--Graph Convolutional Networks (GCN)                                            regression/classification
# 12--Multi-Channel Graph Convolutional Networks (MC-GCN)           regression
# 13--Graph Convolution Long Short-Term Memory (GC-LSTM)          regression
#
# To be continued ...

# Load packages
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits

# Ignore warnings
warnings.filterwarnings('ignore')


# Load dataset
def load_data(data_type='regression', test_size=0.3, seed=1, normalization=True):
    if data_type == 'regression':
        X, y = load_diabetes(return_X_y=True)
    elif data_type == 'classification':
        X, y = load_digits(return_X_y=True)
    else:
        print('You have given a wrong data type.')

    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=test_size, random_state=seed)

    if normalization:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Plot prediction and truth
def plot_pred(y_pred, y_test, title='Title'):
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, 'r')
    plt.grid()
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(title)
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
