# t-distributed Stochastic Neighbor Embedding (t-SNE)
from sklearn.manifold import TSNE
from .main import load_data, scatter


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='classification')

    # Program by package
    print('=====Program by package=====')
    dr = TSNE(n_components=2)
    X_train_pca = dr.fit_transform(X_train)
    X_test_pca = dr.fit_transform(X_test)

    # Plot
    scatter(X_train_pca, y_train.ravel(), 'Train (Package)')
    scatter(X_test_pca, y_test.ravel(), 'Test (Package)')


if __name__ == '__main__':
    mainfunc()
