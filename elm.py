# Extreme Learning Machine (ELM)
import math
import numpy as np
from .main import load_data, plot_pred
from sklearn.base import BaseEstimator, RegressorMixin


# Model
class ElmModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, dim_h=1024, alpha=1.0, direct_link=True, seed=1):
        super(ElmModel, self).__init__()

        # Set seed
        np.random.seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.alpha = alpha
        self.direct_link = direct_link
        self.seed = seed

        # Model creation
        self.w1 = []
        self.w2 = []
        self.std = 1. / math.sqrt(dim_X)
        for i in range(dim_y):
            self.w1.append(np.random.uniform(-self.std, self.std, (dim_X + 1, dim_h)))

    # Add ones
    def add_ones(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    # ReLU
    def relu(self, X):
        return np.maximum(0, X)

    # Train
    def fit(self, X, y):
        X = self.add_ones(X)
        for i in range(self.dim_y):
            h = self.relu(np.matmul(X, self.w1[i]))
            if self.direct_link:
                h = np.concatenate((h, X), axis=1)
            else:
                h = self.add_ones(h)
            self.w2.append(
                np.matmul(np.linalg.inv(np.matmul(h.T, h) + np.eye(h.shape[1]) / self.alpha), np.matmul(h.T, y[:, i])))

        return self

    # Test
    def predict(self, X):
        res_list = []
        X = self.add_ones(X)

        for i in range(self.dim_y):
            h = self.relu(np.matmul(X, self.w1[i]))
            if self.direct_link:
                h = np.concatenate((h, X), axis=1)
            else:
                h = self.add_ones(h)
            res_list.append(np.matmul(h, self.w2[i]).squeeze())
        y = np.stack(res_list, axis=-1)

        return y


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression')

    # Program by myself
    print('=====Program by myself=====')
    mdl = ElmModel(X_train.shape[1], y_train.shape[1], 1024, 0.01).fit(X_train, y_train)
    y_fit = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    plot_pred(y_fit, y_train, 'Train (Myself)')
    plot_pred(y_pred, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
