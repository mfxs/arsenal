# Gaussian Process Regression (GPR)
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from main import load_data, plot_pred
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


# Model
class GprModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, l=0.5, sigma=0.2, optimize=True):
        super(GprModel, self).__init__()
        self.params = {'l': l, 'sigma': sigma}
        self.optimize = optimize

    # Train
    def fit(self, X, y):
        self.X = X
        self.y = y

        def negative_log_likelihood_loss(params):
            self.params['l'], self.params['sigma'] = params[0], params[1]
            Kyy = self.kernel(self.X, self.X) + 1e-8 * np.eye(len(self.X))
            loss = 0.5 * self.y.T.dot(np.linalg.inv(Kyy)).dot(self.y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(
                self.X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params['l'], self.params['sigma']],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)), method='L-BFGS-B')
            self.params['l'], self.params['sigma'] = res.x[0], res.x[1]

        return self

    # Test
    def predict(self, X):
        Kff = self.kernel(self.X, self.X)
        Kyy = self.kernel(X, X)
        Kfy = self.kernel(self.X, X)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.X)))

        mu = Kfy.T.dot(Kff_inv).dot(self.y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)

        return mu, cov

    # Kernel
    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params['sigma'] ** 2 * np.exp(-0.5 / self.params['l'] ** 2 * dist_matrix)


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression')
    l_init, sigma_init = 0.5, 0.2

    # Program by package
    print('=====Program by package=====')
    kernel = ConstantKernel(constant_value=sigma_init, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=l_init,
                                                                                                length_scale_bounds=(
                                                                                                1e-4, 1e4))
    reg = GaussianProcessRegressor(kernel).fit(X_train, y_train)
    y_fit_1 = reg.predict(X_train)
    y_pred_1 = reg.predict(X_test)
    print(
        'Hyper-parameters: l: {:.1f}, sigma: {:.1f}'.format(reg.kernel_.k2.length_scale, reg.kernel_.k1.constant_value))
    print('Fit: {:.4f} Pred: {:.4f}\n'.format(reg.score(X_train, y_train), reg.score(X_test, y_test)))

    # Program by myself
    print('=====Program by myself=====')
    mdl = GprModel(l_init, sigma_init).fit(X_train, y_train)
    y_fit_2, _ = mdl.predict(X_train)
    y_pred_2, _ = mdl.predict(X_test)
    print('Hyper-parameters: l: {:.1f}, sigma: {:.1f}'.format(mdl.params['l'], mdl.params['sigma']))
    print('Fit: {:.4f} Pred: {:.4f}'.format(r2_score(y_train, y_fit_2), r2_score(y_test, y_pred_2)))

    # Plot
    plot_pred(y_fit_1, y_train, 'Train (Package)')
    plot_pred(y_pred_1, y_test, 'Test (Package)')
    plot_pred(y_fit_2, y_train, 'Train (Myself)')
    plot_pred(y_pred_2, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
