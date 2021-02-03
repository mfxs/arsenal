# Least Absolute Shrinkage and Selection Operator (LASSO)
import cvxpy as cp
import numpy as np
from .main import load_data, plot_pred
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV


# Model
class LassoModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, alpha=1):
        super(LassoModel, self).__init__()
        self.alpha = alpha

    # Train
    def fit(self, X, y):
        (m, self.n) = X.shape
        w = cp.Variable(shape=(self.n + 1, 1))
        self.X = np.concatenate((X, np.ones((m, 1))), axis=1)
        self.y = y
        term1 = cp.norm2(self.y - self.X * w) ** 2 / (2 * m)
        term2 = self.alpha * cp.norm1(w)
        problem = cp.Problem(cp.Minimize(term1 + term2))
        problem.solve(solver='SCS')
        self.w = w.value

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
    param = {'alpha': np.logspace(-4, 4, 10000)}

    # Program by package
    print('=====Program by package=====')
    reg = RandomizedSearchCV(Lasso(random_state=1), param, 500, 'r2', iid=True, cv=5, random_state=1).fit(X_train,
                                                                                                          y_train)
    y_fit_1 = reg.predict(X_train)
    y_pred_1 = reg.predict(X_test)
    print('Hyper-parameters: {}'.format(reg.best_params_))
    print('Coefficient: {}'.format(reg.best_estimator_.coef_))
    print('Fit: {:.4f} Pred: {:.4f}\n'.format(reg.score(X_train, y_train), reg.score(X_test, y_test)))

    # Program by myself
    print('=====Program by myself=====')
    mdl = RandomizedSearchCV(LassoModel(), param, 5, 'r2', iid=True, cv=5, random_state=1).fit(X_train, y_train)
    y_fit_2 = mdl.predict(X_train)
    y_pred_2 = mdl.predict(X_test)
    print('Hyper-parameters: {}'.format(mdl.best_params_))
    print('Coefficient: {}'.format(mdl.best_estimator_.w))
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    plot_pred(y_fit_1, y_train, 'Train (Package)')
    plot_pred(y_pred_1, y_test, 'Test (Package)')
    plot_pred(y_fit_2, y_train, 'Train (Myself)')
    plot_pred(y_pred_2, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
