# Graph Convolutional Networks (GCN)
import torch
import torch.nn as nn
import torch.optim as optim
from .main import load_data, plot_pred
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from .packages import MyDataset, GraphConvolution, adjacency_matrix


# Network
class GraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(1024,)):
        super(GraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net_gc = [dim_X, ] + list(gc)
        self.net_fc = [gc[-1], 1]

        # Model creation
        self.gc = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(dim_y):
            self.gc.append(nn.ModuleList())
            for j in range(len(gc)):
                self.gc[-1].append(GraphConvolution(self.net_gc[j], self.net_gc[j + 1]))
            self.fc.append(nn.Linear(self.net_fc[0], self.net_fc[1]))

    # Forward propagation
    def forward(self, X, scale=0.4, gpu=torch.device('cuda:0')):
        adj = adjacency_matrix(X.cpu().numpy(), 'rbf', scale=scale, gpu=gpu)
        res_list = []

        for i in range(self.dim_y):
            feat = X
            for j in self.gc[i]:
                feat = self.act(j(feat, adj))
            feat = self.fc[i](feat)
            res_list.append(feat.squeeze())
        res = torch.stack(res_list, dim=-1)

        return res


# Model
class GcnModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(1024,), scale=0.4, n_epoch=200, batch_size=64, lr=0.001, weight_decay=0.1,
                 step_size=50, gamma=0.5, gpu=torch.device('cuda:0'), seed=1):
        super(GcnModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.gc = gc
        self.scale = scale
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
        self.model = GraphConvolutionalNetworks(dim_X, dim_y, gc).to(gpu)
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
                output = self.model(batch_X, self.scale, self.gpu)
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
            y = self.scaler_y.inverse_transform(self.model(X, self.scale, self.gpu).cpu().numpy())

        return y


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression', normalization=False)

    # Program by myself
    print('=====Program by myself=====')
    mdl = GcnModel(X_train.shape[1], y_train.shape[1], (256,)).fit(X_train, y_train)
    y_fit = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    plot_pred(y_fit, y_train, 'Train (Myself)')
    plot_pred(y_pred, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
