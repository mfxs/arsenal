# Principal Component Analysis (PCA)
import numpy as np
from .main import load_data, scatter
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


# Model
class PcaModel(BaseEstimator, TransformerMixin):

    # Initialization
    def __init__(self):
        super(PcaModel, self).__init__()

    # Fit & Transform
    def fit_transform(self, X):
        w, v = np.linalg.eig(np.matmul(X.T, X))
        self.load = v
        self.eigen_value = w

        return np.matmul(X, v)

    # Transform
    def transform(self, X):
        return np.matmul(X, self.load)


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='classification')

    # Program by package
    print('=====Program by package=====')
    dr = PCA()
    X_train_pca_1 = dr.fit_transform(X_train)
    X_test_pca_1 = dr.transform(X_test)
    print('Variance: {}\n'.format(np.cumsum(dr.singular_values_ ** 2 / sum(dr.singular_values_ ** 2))))

    # Program by myself
    print('=====Program by myself=====')
    mdl = PcaModel()
    X_train_pca_2 = mdl.fit_transform(X_train)
    X_test_pca_2 = mdl.transform(X_test)
    print('Variance: {}'.format(np.cumsum(mdl.eigen_value / sum(mdl.eigen_value))))

    # Plot
    scatter(X_train_pca_1, y_train.ravel(), 'Train (Package)')
    scatter(X_test_pca_1, y_test.ravel(), 'Test (Package)')
    scatter(X_train_pca_2, y_train.ravel(), 'Train (Myself)')
    scatter(X_test_pca_2, y_test.ravel(), 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
