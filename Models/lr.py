# Logistic Regression (LR)
import numpy as np
from main import load_data, confusion
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV


# Model
class LrModel(BaseEstimator, ClassifierMixin):

    # Initialization
    def __init__(self, C=1000, lr=0.001, max_iter=200, seed=1):
        super(LrModel, self).__init__()
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        np.random.seed(seed)

    # Train
    def fit(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.classes = np.unique(y)
        self.n_classes = self.classes.shape[0]
        if self.n_classes == 2:
            self.w = np.random.uniform(-1, 1, (X.shape[1], 1))
        else:
            self.w = np.random.uniform(-1, 1, (X.shape[1], self.n_classes))
        for i in range(self.w.shape[1]):
            y_temp = np.zeros(y.shape[0])
            y_temp[y == self.classes[i]] = 1
            for j in range(self.max_iter):
                grad = np.matmul(X.T, 1 / (1 + np.exp(-np.matmul(X, self.w[:, i]))) - y_temp)
                self.w[:, i] = self.w[:, i] - self.lr * (self.C * grad / X.shape[0] + 2 * self.w[:, i])

        return self

    # Test
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_pred = 1 / (1 + np.exp(-np.matmul(X, self.w)))
        y_pred = self.classes[np.argmax(y_pred, axis=1)]

        return y_pred


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='classification')
    param = {'C': np.logspace(-4, 4, 10000)}

    # Program by package
    print('=====Program by package=====')
    clf = RandomizedSearchCV(LogisticRegression(solver='liblinear', multi_class='ovr', random_state=1), param, 10,
                             'accuracy', iid=True, cv=5, random_state=1).fit(X_train, y_train)
    y_fit_1 = clf.predict(X_train)
    y_pred_1 = clf.predict(X_test)
    print('Hyper-parameters: {}'.format(clf.best_params_))
    print('Fit: {:.4f} Pred: {:.4f}\n'.format(clf.score(X_train, y_train), clf.score(X_test, y_test)))

    # Program by myself
    print('=====Program by myself=====')
    mdl = RandomizedSearchCV(LrModel(), param, 10, 'accuracy', iid=True, cv=5, random_state=1).fit(X_train,
                                                                                                   y_train.ravel())
    y_fit_2 = mdl.predict(X_train)
    y_pred_2 = mdl.predict(X_test)
    print('Hyper-parameters: {}'.format(mdl.best_params_))
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    confusion(y_fit_1, y_train, 'Train (Package)')
    confusion(y_pred_1, y_test, 'Test (Package)')
    confusion(y_fit_2, y_train, 'Train (Myself)')
    confusion(y_pred_2, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
