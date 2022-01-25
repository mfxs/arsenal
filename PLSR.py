# PLSR: Partial Least Square Regression
from main import *
from sklearn.cross_decomposition import PLSRegression


# Model
class PlsrModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, n_components=2):
        super(PlsrModel, self).__init__()
        self.n_components = n_components
        self.scaler = StandardScaler()

    # Train
    def fit(self, X, y):
        self.X = X
        self.y = self.scaler.fit_transform(y)
        E0 = self.X
        F0 = self.y

        # Single projection direction
        self.w = []

        # Single projection component
        self.t = []

        # Regression coefficient for E
        self.p = []
        for i in range(self.n_components):
            eig_val, eig_vec = np.linalg.eig(np.matmul(np.matmul(E0.T, F0), np.matmul(F0.T, E0)))
            w1 = np.real(np.atleast_2d(eig_vec[:, np.argmax(eig_val)]).T)
            self.w.append(w1)
            t1 = np.matmul(E0, w1)
            self.t.append(t1)
            p1 = np.matmul(E0.T, t1) / np.linalg.norm(t1) ** 2
            self.p.append(p1)
            r1 = np.matmul(F0.T, t1) / np.linalg.norm(t1) ** 2
            E0 = E0 - np.matmul(t1, p1.T)
            F0 = F0 - np.matmul(t1, r1.T)
        self.w = np.stack(self.w, axis=1).squeeze()
        self.t = np.stack(self.t, axis=1).squeeze()
        self.p = np.stack(self.p, axis=1).squeeze()

        # Overall projection direction
        self.r = np.matmul(self.w, np.linalg.inv(np.matmul(self.p.T, self.w)))

        # Regression coefficient for y
        self.theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.t.T, self.t)), self.t.T), self.y)

        return self

    # Test
    def predict(self, X):
        y_pred = self.scaler.inverse_transform(np.matmul(np.matmul(X, self.r), self.theta))

        return y_pred


# Main function
def mainfunc():
    seed = 123

    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression', seed=seed)
    param = {'n_components': range(2, X_train.shape[1] + 1)}

    # Program by package
    print('=====Program by package=====')
    reg = GridSearchCV(PLSRegression(), param, 'r2', iid=True, cv=5).fit(X_train, y_train)
    y_fit_1 = reg.predict(X_train)
    y_pred_1 = reg.predict(X_test)
    print('Hyper-parameters: {}'.format(reg.best_params_))
    print('Fit: {:.4f} Pred: {:.4f}\n'.format(reg.score(X_train, y_train), reg.score(X_test, y_test)))

    # Program by myself
    print('=====Program by myself=====')
    mdl = GridSearchCV(PlsrModel(), param, 'r2', iid=True, cv=5).fit(X_train, y_train)
    y_fit_2 = mdl.predict(X_train)
    y_pred_2 = mdl.predict(X_test)
    print('Hyper-parameters: {}'.format(mdl.best_params_))
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    plot_pred(y_fit_1, y_train, 'Train (Package)')
    plot_pred(y_pred_1, y_test, 'Test (Package)')
    plot_pred(y_fit_2, y_train, 'Train (Myself)')
    plot_pred(y_pred_2, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
