from Base.packages import *
from Models import OLS, RR, LASSO, PLSR, GPR, ELM, MCGCN, GCLSTM

model_myself = {
    'OLS': OLS.OlsModel(),
    'RR': RR.RrModel(),
    'LASSO': LASSO.LassoModel(),
    'PLSR': PLSR.PlsrModel(),
    'GPR': GPR.GprModel(),
    'ELM': ELM.ElmModel(),
    'MCGCN': MCGCN.McgcnModel(),
    'GCLSTM': GCLSTM.GclstmModel()
}

model_package = {
    'OLS': OLS.LinearRegression(),
    'RR': RR.Ridge(),
    'LASSO': LASSO.Lasso(),
    'PLSR': PLSR.PLSRegression(),
    'GPR': GPR.GaussianProcessRegressor(),
    'ELM': None,
    'MCGCN': None,
    'GCLSTM': None
}

hyper_params = {
    'OLS': None,
    'RR': {'alpha': np.logspace(-4, 4, 10000)},
    'LASSO': {'alpha': np.logspace(-4, 4, 10000)},
    'PLSR': {'n_components': range(2, 11)},
    'GPR': {'l': np.linspace(0.1, 1.0, 10), 'sigma': np.linspace(0.1, 1.0, 10)},
    'ELM': {'dim_h': [1024, 512, 256], 'alpha': np.logspace(-4, 4, 10000)},
    'MCGCN': None,
    'GCLSTM': None
}

hpo = {
    'GS': {'cv': 5},
    'RS': {'cv': 5, 'n_iter': 100}
}
