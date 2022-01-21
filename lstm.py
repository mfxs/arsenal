# Long Short-Term Memory (LSTM)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from packages import MyDataset
from sklearn.metrics import r2_score
from main import load_data, plot_pred
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin


# Network
class LongShortTermMemory(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), mode='mvm'):
        super(LongShortTermMemory, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net_lstm = [dim_X, ] + list(lstm)
        self.net_fc = [lstm[-1], 1]
        self.mode = mode

        # Model creation
        self.lstm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(dim_y):
            self.lstm.append(nn.ModuleList())
            for j in range(len(lstm)):
                self.lstm[-1].append(nn.LSTM(self.net_lstm[j], self.net_lstm[j + 1], batch_first=True))
            self.fc.append(nn.Linear(self.net_fc[0], self.net_fc[1]))

    # Forward propagation
    def forward(self, X):
        res_list = []

        for i in range(self.dim_y):
            feat = X
            for j in self.lstm[i]:
                feat, _ = j(feat)
            if self.mode == 'mvm':
                feat = self.fc[i](feat)
            elif self.mode == 'mvo':
                feat = self.fc[i](_[0])
            else:
                raise Exception('Wrong mode selection.')
            res_list.append(feat.squeeze())
        res = torch.stack(res_list, dim=-1)

        return res


# Model
class LstmModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), mode='mvm', input='2D', seq_len=30, n_epoch=200, batch_size=64,
                 lr=0.001, weight_decay=0.1, step_size=50, gamma=0.5, gpu=0, seed=1):
        super(LstmModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.lstm = lstm
        self.mode = mode if input == '2D' else 'mvo'
        self.input = input
        self.seq_len = seq_len
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
        self.model = LongShortTermMemory(dim_X, dim_y, lstm, self.mode).to(gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        if self.input == '2D':
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y)
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
        elif self.input == '3D':
            self.seq_len = X.shape[1]
            X_3d = X
            dataset_y = self.scaler_y.fit_transform(y)
        else:
            raise Exception('Wrong input selection.')
        dataset = MyDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.gpu),
                            torch.tensor(dataset_y, dtype=torch.float32, device=self.gpu))
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
        if self.input == '2D':
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
        elif self.input == '3D':
            X_3d = torch.tensor(X, dtype=torch.float32, device=self.gpu)
        else:
            raise Exception('Wrong input selection.')
        self.model.eval()
        with torch.no_grad():
            y = self.scaler_y.inverse_transform(self.model(X_3d).cpu().numpy())

        return y


# Main function
def mainfunc():
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_type='regression', normalization=False)

    # Program by myself
    print('=====Program by myself=====')
    mdl = LstmModel(X_train.shape[1], y_train.shape[1], (256,), 'mvm').fit(X_train, y_train)
    y_fit = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)
    if mdl.mode == 'mvo':
        print('Fit: {:.4f} Pred: {:.4f}'.format(r2_score(y_train[mdl.seq_len - 1:], y_fit),
                                                r2_score(y_test[mdl.seq_len - 1:], y_pred)))
    else:
        print('Fit: {:.4f} Pred: {:.4f}'.format(mdl.score(X_train, y_train), mdl.score(X_test, y_test)))

    # Plot
    if mdl.mode == 'mvo':
        plot_pred(y_fit, y_train[mdl.seq_len - 1:], 'Train (Myself)')
        plot_pred(y_pred, y_test[mdl.seq_len - 1:], 'Test (Myself)')
    else:
        plot_pred(y_fit, y_train, 'Train (Myself)')
        plot_pred(y_pred, y_test, 'Test (Myself)')


if __name__ == '__main__':
    mainfunc()
