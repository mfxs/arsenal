# Ordinary Least Square (OLS)
import numpy as np
from .main import load_data, plot_pred
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin


# Model
class OlsModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self):
        super(OlsModel, self).__init__()

    # Train
    def fit(self, X, y):
        m = X.shape[0]
        self.X = np.concatenate((X, np.ones((m, 1))), axis=1)
        self.y = y
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.y)

        return self

    # Test
    def predict(self, X):
        m = X.shape[0]
        X = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_pred = np.matmul(X, self.w)

        return y_pred


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression')

    # Program by package
    print('=====Program by package=====')
    reg = LinearRegression().fit(X_train, y_train)
    y_fit_1 = reg.predict(X_train)
    y_pred_1 = reg.predict(X_test)
    print('Coefficient: {}'.format(reg.coef_))
    print('Fit: {:.4f} Pred: {:.4f}\n'.format(reg.score(X_train, y_train), reg.score(X_test, y_test)))

    # Program by myself
    print('=====Program by myself=====')
    mdl = OlsModel().fit(X_train, y_train)
    y_fit_2 = mdl.predict(X_train)
    y_pred_2 = mdl.predict(X_test)
    print('Coefficient: {}'.format(mdl.w))
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    plot_pred(y_fit_1, y_train, 'Train (Package)')
    plot_pred(y_pred_1, y_test, 'Test (Package)')
    plot_pred(y_fit_2, y_train, 'Train (Myself)')
    plot_pred(y_pred_2, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
