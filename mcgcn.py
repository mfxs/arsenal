# Multi-Channel Graph Convolutional Networks (MC-GCN)
import torch
import numpy as np
from torch import nn
from torch import optim
from .main import load_data, plot_pred
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from .packages import MyDataset, GraphConvolution, adjacency_matrix


# Network
class MultiChannelGraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, in_fc=(1024,), gc=(256,), out_fc=(256, 256)):
        super(MultiChannelGraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net_in_fc = [dim_X, ] + list(in_fc)
        self.net_gc = [in_fc[-1], ] + list(gc)
        self.net_out_fc = [gc[-1], ] + list(out_fc) + [1, ]

        # Input FC & Output FC
        self.in_fc = nn.ModuleList()
        self.out_fc = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(dim_y):
            self.in_fc.append(nn.ModuleList())
            for j in range(len(in_fc)):
                self.in_fc[-1].append(nn.Sequential(nn.Linear(self.net_in_fc[j], self.net_in_fc[j + 1]), nn.ReLU()))
            self.out_fc.append(nn.ModuleList())
            for j in range(len(out_fc)):
                self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]), nn.ReLU()))
            self.out_fc[-1].append(nn.Linear(self.net_out_fc[-2], self.net_out_fc[-1]))

        # GC
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.net_gc[i], self.net_gc[i + 1]))

    # Forward propagation
    def forward(self, X, adj):
        feat_list = []
        res_list = []

        # Input FC
        for i in range(self.dim_y):
            feat = X
            for j in self.in_fc[i]:
                feat = j(feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)

        # GC
        for gc in self.gc:
            feat = gc(feat, adj)
            feat = self.act(feat)

        # Output FC
        for i in range(self.dim_y):
            res = feat[:, i, :]
            for j in self.out_fc[i]:
                res = j(res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)

        return res


# Model
class McgcnModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, in_fc=(1024,), gc=(256,), out_fc=(256, 256), graph_reg=0.05, self_con=0.2,
                 n_epoch=200, batch_size=64, lr=0.001, weight_decay=0.1, step_size=50, gamma=0.5,
                 gpu=torch.device('cuda:0'), seed=1):
        super(McgcnModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.in_fc = in_fc
        self.gc = gc
        self.out_fc = out_fc
        self.graph_reg = graph_reg
        self.self_con = self_con
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
        self.model = MultiChannelGraphConvolutionalNetworks(dim_X, dim_y, in_fc, gc, out_fc).to(gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        self.adj = adjacency_matrix(y, 'sc', self.graph_reg, self.self_con, gpu=self.gpu)
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y, dtype=torch.float32, device=self.gpu))
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X, self.adj)
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
            y = self.scaler_y.inverse_transform(self.model(X, self.adj).cpu().numpy())

        return y


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression', normalization=False)
    y_train = np.concatenate((y_train, y_train ** 2), axis=1)
    y_test = np.concatenate((y_test, y_test ** 2), axis=1)

    # Program by myself
    print('=====Program by myself=====')
    mdl = McgcnModel(X_train.shape[1], y_train.shape[1], (1024,), (256,), (256,)).fit(X_train, y_train)
    y_fit = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)
    print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    for i in range(y_train.shape[1]):
        plot_pred(y_fit[:, i], y_train[:, i], 'Train (Myself, Variable {})'.format(i + 1))
        plot_pred(y_pred[:, i], y_test[:, i], 'Test (Myself, Variable {})'.format(i + 1))


if __name__ == '__main__':
    mainfunc()
