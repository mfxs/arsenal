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

# Packages
from Base.plot import *
from Base.utils import *
from Base.hyper_params import *

# Ignore warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser('Arsenal for machine learning')
parser.add_argument('-prob', type=str, default='regression')
parser.add_argument('-model', type=str, default='GCLSTM')
parser.add_argument('-myself', type=bool, default=True)
parser.add_argument('-multi_y', type=bool, default=True)
parser.add_argument('-hpo', type=bool, default=False)
parser.add_argument('-hpo_method', type=str, default='RS')
parser.add_argument('-seed', type=int, default=123)
args = parser.parse_args()


def mainfunc():
    # Load data
    print('=====Loading data=====')
    X_train, X_test, y_train, y_test = load_data(args.prob, seed=args.seed)
    if args.multi_y:
        y_train = np.concatenate((y_train, y_train ** 2, y_train ** 3), axis=1)
        y_test = np.concatenate((y_test, y_test ** 2, y_test ** 3), axis=1)
    print('Dataset for {} problem has been loaded'.format(args.prob) + '\n')

    # Model construction
    print('=====Constructing model=====')
    print('{} model is selected'.format(args.model))
    if args.myself or model_package[args.model] is None:
        model = model_myself[args.model]
        print('Model by myself')
    else:
        model = model_package[args.model]
        print('Model by package')
    if args.hpo and hyper_params[args.model]:
        if args.hpo_method == 'GS':
            model = GridSearchCV(model, hyper_params[args.model], cv=hpo['GS']['cv'])
            print('Grid search for hpo')
        elif args.hpo_method == 'RS':
            model = RandomizedSearchCV(model, hyper_params[args.model], cv=hpo['RS']['cv'], n_iter=hpo['RS']['cv'])
            print('Random search for hpo')
        else:
            raise Exception('Wrong method for hpo')
        model.fit(X_train, y_train)
        print('Best hyper-params: {}'.format(model.best_params_))
    else:
        print('Default params for modelling')
        model.fit(X_train, y_train)
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test)
    print('Modelling is finished\n')

    # Plot result
    print('=====Evaluating model=====')
    if args.model in ['LSTM', 'GCLSTM'] and model.args['mode'] == 'mvo':
        y_train = y_train[model.args['seq_len'] - 1:]
        y_test = y_test[model.args['seq_len'] - 1:]
    if args.prob == 'regression':
        pred_curve(y_train, y_fit, 'Train')
        pred_curve(y_test, y_pred, 'Test')
        pred_scatter(y_train, y_fit, 'Train')
        pred_scatter(y_test, y_pred, 'Test')
    elif args.prob == 'classification':
        confusion(y_train, y_fit, 'Train')
        confusion(y_test, y_pred, 'Test')
    print('Evaluating is finished')


if __name__ == '__main__':
    mainfunc()
