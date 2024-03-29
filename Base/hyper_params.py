# Some hyper-parameters for modelling
from arsenal.Base.packages import *
from arsenal.Models import OLS, RR, LASSO, PLSR, GPR, ELM, MCGCN, GCLSTM, LR, FCN, LSTM, GCN, PCA, tSNE, AE, VAE

# Model implemented by myself
model_myself = {
    'regression': {
        'OLS': OLS.OlsModel(),
        'RR': RR.RrModel(),
        'LASSO': LASSO.LassoModel(),
        'PLSR': PLSR.PlsrModel(),
        'GPR': GPR.GprModel(),
        'ELM': ELM.ElmModel(),
        'MCGCN': MCGCN.McgcnModel(),
        'GCLSTM': GCLSTM.GclstmModel(),
        'FCN': FCN.FcnModel(prob='regression'),
        'LSTM': LSTM.LstmModel(prob='regression'),
        'GCN': GCN.GcnModel(prob='regression')
    },
    'classification': {
        'LR': LR.LrModel(),
        'FCN': FCN.FcnModel(prob='classification'),
        'LSTM': LSTM.LstmModel(prob='classification'),
        'GCN': GCN.GcnModel(prob='classification')
    },
    'dimensionality-reduction': {
        'PCA': PCA.PcaModel(),
        'AE': AE.AeModel(),
        'VAE': VAE.VaeModel()
    }
}

# Model implemented by package
model_package = {
    'regression': {
        'OLS': OLS.LinearRegression(),
        'RR': RR.Ridge(),
        'LASSO': LASSO.Lasso(),
        'PLSR': PLSR.PLSRegression(),
        'GPR': GPR.GaussianProcessRegressor(),
        'FCN': FCN.MLPRegressor()
    },
    'classification': {
        'LR': LR.LogisticRegression(),
        'FCN': FCN.MLPClassifier()
    },
    'dimensionality-reduction': {
        'PCA': PCA.PCA(),
        'tSNE': tSNE.TSNE()
    }
}

# Model hyper-parameters
hyper_params = {
    'RR': {'alpha': np.logspace(-4, 4, 10000)},
    'LASSO': {'alpha': np.logspace(-4, 4, 10000)},
    'PLSR': {'n_components': range(2, 11)},
    'GPR': {'l': np.linspace(0.1, 1.0, 10), 'sigma': np.linspace(0.1, 1.0, 10)},
    'ELM': {'dim_h': [1024, 512, 256], 'alpha': np.logspace(-4, 4, 10000)},
    'LR': {'C': np.logspace(-4, 4, 10000)},
    'MCGCN': {'in_fc': ((1024,), (512,), (256,))},
    'GCLSTM': {'lstm': ((1024,), (512,), (256,))},
    'FCN': {'hidden_layer_sizes': ((1024,), (512,), (256,), (128,), (1024, 512), (512, 256), (256, 128))},
    'LSTM': {'lstm': ((1024,), (512,), (256,), (128,), (1024, 512), (512, 256), (256, 128))},
    'GCN': {'gc': ((1024,), (512,), (256,), (128,), (1024, 512), (512, 256), (256, 128))}
}

# HPO hyper-parameters
hpo = {
    'GS': {'cv': 5},
    'RS': {'cv': 5, 'n_iter': 100}
}
