from Base.packages import *
from Models import OLS, RR, LASSO, PLSR, GPR, ELM, MCGCN, GCLSTM, LR, FCN, LSTM

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
        'LSTM': LSTM.LstmModel(prob='regression')
    },
    'classification': {
        'LR': LR.LrModel(),
        'FCN': FCN.FcnModel(prob='classification'),
        'LSTM': LSTM.LstmModel(prob='classification')
    },
    'dimensionality reduction': {}
}

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
    'dimensionality reduction': {}
}

hyper_params = {
    'RR': {'alpha': np.logspace(-4, 4, 10000)},
    'LASSO': {'alpha': np.logspace(-4, 4, 10000)},
    'PLSR': {'n_components': range(2, 11)},
    'GPR': {'l': np.linspace(0.1, 1.0, 10), 'sigma': np.linspace(0.1, 1.0, 10)},
    'ELM': {'dim_h': [1024, 512, 256], 'alpha': np.logspace(-4, 4, 10000)},
    'LR': {'C': np.logspace(-4, 4, 10000)},
    'FCN': {'hidden_layers': ((1024,), (512,), (256,), (128,), (1024, 512), (512, 256), (256, 128))},
    'LSTM': {'hidden_layers': ((1024,), (512,), (256,), (128,), (1024, 512), (512, 256), (256, 128))}
}

hpo = {
    'GS': {'cv': 5},
    'RS': {'cv': 5, 'n_iter': 100}
}
