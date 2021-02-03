# Fully Connected Networks (FCN)
import torch
import torch.nn as nn
import torch.optim as optim
from .packages import MyDataset
from .main import load_data, plot_pred
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin


# Network
class FullyConnectedNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, hidden_layers=(256,)):
        super(FullyConnectedNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.network_structure = [dim_X, ] + list(hidden_layers) + [1, ]

        # Model creation
        self.net = nn.ModuleList()
        for i in range(dim_y):
            self.net.append(nn.ModuleList())
            for j in range(len(hidden_layers)):
                self.net[-1].append(
                    nn.Sequential(nn.Linear(self.network_structure[j], self.network_structure[j + 1]), nn.ReLU()))
            self.net[-1].append(nn.Linear(self.network_structure[-2], self.network_structure[-1]))

    # Forward propagation
    def forward(self, X):
        res_list = []

        for i in range(self.dim_y):
            feat = X
            for j in self.net[i]:
                feat = j(feat)
            res_list.append(feat.squeeze())

        res = torch.stack(res_list, dim=1)

        return res


# Model
class FcnModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, hidden_layers=(256,), n_epoch=200, batch_size=64, lr=0.001, weight_decay=0.01,
                 step_size=50, gamma=0.5, gpu=torch.device('cuda:0'), seed=1):
        super(FcnModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.hidden_layers = hidden_layers
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.gpu = gpu
        self.seed = seed

        # Initialize scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model creation
        self.loss_hist = []
        self.model = FullyConnectedNetworks(dim_X, dim_y, hidden_layers).to(gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y, dtype=torch.float32, device=self.gpu))
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # Test
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        with torch.no_grad():
            y = self.scaler_y.inverse_transform(self.model(X).cpu().numpy())

        return y


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression', normalization=False)

    # Program by package
    print('=====Program by package=====')
    reg = MLPRegressor((512, 256), random_state=1).fit(X_train, y_train)
    y_fit_1 = reg.predict(X_train)
    y_pred_1 = reg.predict(X_test)
    print('Fit: {:.4f} Pred: {:.4f}\n'.format(reg.score(X_train, y_train), reg.score(X_test, y_test)))

    # Program by myself
    print('=====Program by myself=====')
    mdl = FcnModel(X_train.shape[1], y_train.shape[1], (512, 256)).fit(X_train, y_train)
    y_fit_2 = mdl.predict(X_train)
    y_pred_2 = mdl.predict(X_test)
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    plot_pred(y_fit_1, y_train, 'Train (Package)')
    plot_pred(y_pred_1, y_test, 'Test (Package)')
    plot_pred(y_fit_2, y_train, 'Train (Myself)')
    plot_pred(y_pred_2, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
