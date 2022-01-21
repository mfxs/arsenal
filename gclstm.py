# Graph Convolution Long Short-Term Memory (GC-LSTM)
import torch
import numpy as np
from torch import nn
from torch import optim
from sklearn.metrics import r2_score
from main import load_data, plot_pred
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from packages import MyDataset, GraphConvolution, adjacency_matrix


# Network
class GraphConvolutionLongShortTermMemory(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), gc=(256,), fc=(256, 256), mode='mvm'):
        super(GraphConvolutionLongShortTermMemory, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net_lstm = [dim_X, ] + list(lstm)
        self.net_gc = [lstm[-1], ] + list(gc)
        self.net_fc = [gc[-1], ] + list(fc) + [1, ]
        self.mode = mode

        # LSTM & FC
        self.lstm = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(dim_y):
            self.lstm.append(nn.ModuleList())
            for j in range(len(lstm)):
                self.lstm[-1].append(nn.LSTM(self.net_lstm[j], self.net_lstm[j + 1], batch_first=True))
            self.fc.append(nn.ModuleList())
            for j in range(len(fc)):
                self.fc[-1].append(nn.Sequential(nn.Linear(self.net_fc[j], self.net_fc[j + 1]), nn.ReLU()))
            self.fc[-1].append(nn.Linear(self.net_fc[-2], self.net_fc[-1]))

        # GC
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.net_gc[i], self.net_gc[i + 1]))

    # Forward propagation
    def forward(self, X, adj):
        feat_list = []
        res_list = []

        # LSTM
        for i in range(self.dim_y):
            feat = X
            for j in self.lstm[i]:
                feat, _ = j(feat)
            if self.mode == 'mvm':
                feat_list.append(feat)
            elif self.mode == 'mvo':
                feat_list.append(_[0])
            else:
                raise Exception('Wrong mode selection.')
        feat = torch.stack(feat_list, dim=-2)

        # GC
        for gc in self.gc:
            feat = gc(feat, adj)
            feat = self.act(feat)

        # FC
        for i in range(self.dim_y):
            res = feat[:, :, i, :]
            for j in self.fc[i]:
                res = j(res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=-1)

        return res


# Model
class GclstmModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), gc=(256,), fc=(256, 256), mode='mvm', seq_len=30, graph_reg=0.05,
                 self_con=0.2, n_epoch=200, batch_size=64, lr=0.001, weight_decay=0.1, step_size=50, gamma=0.5, gpu=0,
                 seed=1):
        super(GclstmModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.lstm = lstm
        self.gc = gc
        self.fc = fc
        self.mode = mode
        self.seq_len = seq_len
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
        self.model = GraphConvolutionLongShortTermMemory(dim_X, dim_y, lstm, gc, fc, mode).to(gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        self.adj = adjacency_matrix(y, 'sc', self.graph_reg, self.self_con, gpu=self.gpu)
        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_len + 1):
            X_3d.append(X[i:i + self.seq_len, :])
            y_3d.append(y[i:i + self.seq_len, :])
        X_3d = np.stack(X_3d)
        y_3d = np.stack(y_3d)
        if self.mode == 'mvm':
            dataset_y = y_3d
        elif self.mode == 'mvo':
            dataset_y = y[self.seq_len - 1:]
        else:
            raise Exception('Wrong mode selection.')
        dataset = MyDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.gpu),
                            torch.tensor(dataset_y, dtype=torch.float32, device=self.gpu))
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
        X = self.scaler_X.transform(X)
        if self.mode == 'mvm':
            X_3d = torch.tensor(X, dtype=torch.float32, device=self.gpu).unsqueeze(0)
        elif self.mode == 'mvo':
            X_3d = []
            for i in range(X.shape[0] - self.seq_len + 1):
                X_3d.append(X[i:i + self.seq_len, :])
            X_3d = torch.tensor(np.stack(X_3d), dtype=torch.float32, device=self.gpu)
        else:
            raise Exception('Wrong mode selection.')
        self.model.eval()
        with torch.no_grad():
            y = self.scaler_y.inverse_transform(self.model(X_3d, self.adj).cpu().numpy())

        return y


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression', normalization=False)
    y_train = np.concatenate((y_train, y_train ** 2), axis=1)
    y_test = np.concatenate((y_test, y_test ** 2), axis=1)

    # Program by myself
    print('=====Program by myself=====')
    mdl = GclstmModel(X_train.shape[1], y_train.shape[1], (1024,), (256,), (256,), 'mvo').fit(X_train, y_train)
    y_fit = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)
    if mdl.mode == 'mvo':
        print('Fit: {:.4f} Pred: {:.4f}'.format(r2_score(y_train[mdl.seq_len - 1:], y_fit),
                                                r2_score(y_test[mdl.seq_len - 1:], y_pred)))
    else:
        print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    if mdl.mode == 'mvo':
        for i in range(y_train.shape[1]):
            plot_pred(y_fit[:, i], y_train[mdl.seq_len - 1:, i], 'Train (Myself, Variable {})'.format(i + 1))
            plot_pred(y_pred[:, i], y_test[mdl.seq_len - 1:, i], 'Test (Myself, Variable {})'.format(i + 1))
    else:
        for i in range(y_train.shape[1]):
            plot_pred(y_fit[:, i], y_train[:, i], 'Train (Myself, Variable {})'.format(i + 1))
            plot_pred(y_pred[:, i], y_test[:, i], 'Test (Myself, Variable {})'.format(i + 1))


if __name__ == '__main__':
    mainfunc()
