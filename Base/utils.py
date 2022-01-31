# Some functions and classes for utilization
from Base.packages import *


# Load dataset
def load_data(data_type='regression', test_size=0.3, seed=123, normalization=None):
    # Dataset type
    if data_type == 'regression':
        X, y = load_diabetes(return_X_y=True)
    elif data_type in ['classification', 'dimensionality-reduction']:
        X, y = load_digits(return_X_y=True)
    else:
        raise Exception('Wrong data type.')

    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Adjacency matrix
def adjacency_matrix(X, mode='sc', graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1, gpu=0):
    # RBF kernel function
    if mode == 'rbf':
        kernel = RBF(length_scale=scale)
        A = kernel(X, X)

    # Pearson correlation coefficient
    elif mode == 'pearson':
        A = np.corrcoef(X.T)

    # Sparse coding
    elif mode == 'sc':
        A = cp.Variable((X.shape[1], X.shape[1]))
        term1 = cp.norm(X * A - X, p='fro')
        term2 = cp.norm1(A)
        constraints = []
        for i in range(X.shape[1]):
            constraints.append(A[i, i] == 0)
            for j in range(X.shape[1]):
                constraints.append(A[i, j] >= 0)
        constraints.append(A == A.T)
        objective = cp.Minimize(term1 + graph_reg * term2)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        A = A.value
        A = A + self_con * np.eye(X.shape[1])

    else:
        raise Exception('Wrong mode selection.')

    # Omit small values
    A[np.abs(A) < epsilon] = 0

    # Normalization
    D = np.diag(np.sum(A, axis=1) ** (-0.5))
    A = np.matmul(np.matmul(D, A), D)
    A = torch.tensor(A, dtype=torch.float32).cuda(gpu)

    return A


# Transform to 3d data
def transform3d(X, y, seq_len=30, mode='mvo'):
    X_3d = []
    y_3d = []
    for i in range(X.shape[0] - seq_len + 1):
        X_3d.append(X[i:i + seq_len])
        y_3d.append(y[i:i + seq_len])
    X_3d = np.stack(X_3d)
    if mode == 'mvo':
        y_3d = y[seq_len - 1:]
    elif mode == 'mvm':
        y_3d = np.stack(y_3d)
    else:
        raise Exception('Wrong mode selection.')

    return X_3d, y_3d


# MyDataset
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label, prob='regression', gpu=0):
        super(MyDataset, self).__init__()
        self.gpu = gpu
        self.data = self.__transform__(data)
        self.label = self.__transform__(label, prob)

    # Transform
    def __transform__(self, data, prob='regression'):
        if prob in ['regression', 'dimensionality-reduction']:
            return torch.tensor(data, dtype=torch.float32).cuda(self.gpu)
        else:
            return torch.tensor(data, dtype=torch.long).cuda(self.gpu)

    # Get item
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    # Get length
    def __len__(self):
        return self.data.shape[0]


# Graph convolution
class GraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj=None):
        super(GraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.weight = Parameter(torch.FloatTensor(dim_X, dim_y))
        self.reset_parameters()

    # Forward propagation
    def forward(self, X, adj=None):
        if self.adj is not None:
            res = torch.matmul(self.adj, torch.matmul(X, self.weight))
        elif adj is not None:
            res = torch.matmul(adj, torch.matmul(X, self.weight))
        else:
            raise Exception('No adjacency matrix available.')
        return res

    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)


# Neural network
class NeuralNetwork(BaseEstimator):

    # Initialization
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.scaler = MinMaxScaler()
        self.args = {
            'prob': 'regression',
            'n_epoch': 200,
            'batch_size': 64,
            'lr': 0.001,
            'weight_decay': 0.01,
            'step_size': 50,
            'gamma': 0.5,
            'gpu': 0,
            'seed': 123
        }

    # Data creation
    def data_create(self, X, y, adj=False):
        self.dim_X = X.shape[-1]
        if self.args['prob'] == 'regression':
            y = self.scaler.fit_transform(y)
            self.dim_y = y.shape[-1]
        elif self.args['prob'] == 'classification':
            self.dim_y = np.unique(y).shape[0]
        if adj:
            self.adj = adjacency_matrix(y, self.args['adj_mode'], self.args['graph_reg'], self.args['self_con'],
                                        self.args['scale'], self.args['epsilon'], gpu=self.args['gpu'])
        if 'mode' in self.args.keys():
            self.X, self.y = transform3d(X, y, self.args['seq_len'], self.args['mode'])
        else:
            self.X = X
            self.y = y

        self.dataset = MyDataset(self.X, self.y, self.args['prob'], self.args['gpu'])
        self.dataloader = DataLoader(self.dataset, batch_size=self.args['batch_size'], shuffle=True)

    # Model creation
    def model_create(self, loss='MSE'):
        self.loss_hist = np.zeros(self.args['n_epoch'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args['step_size'], self.args['gamma'])
        if loss == 'MSE':
            self.criterion = nn.MSELoss(reduction='sum')
        elif loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise Exception('Wrong loss function.')

    # Model training
    def training(self):
        self.model.train()
        for i in range(self.args['n_epoch']):
            start = time.time()
            for batch_X, batch_y in self.dataloader:
                if self.args['prob'] == 'classification':
                    batch_y = batch_y.view(-1)
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[i] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            end = time.time()
            print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}s'.format(i + 1, self.loss_hist[i], end - start))

    # Test
    def predict(self, X):
        if 'mode' in self.args.keys():
            if self.args['mode'] == 'mvm':
                X, _ = transform3d(X, X, X.shape[0])
            elif self.args['mode'] == 'mvo':
                X, _ = transform3d(X, X, self.args['seq_len'])
        X = torch.tensor(X, dtype=torch.float32).cuda(self.args['gpu'])
        self.model.eval()
        with torch.no_grad():
            if self.args['prob'] == 'regression':
                y = self.scaler.inverse_transform(self.model(X).cpu().numpy())
            else:
                y = np.argmax(self.model(X).cpu().numpy(), 1)

        return y

    # Score
    def score(self, X, y, index='r2'):
        y_pred = self.predict(X)
        if self.args['prob'] == 'regression':
            if index == 'r2':
                r2 = r2_score(y, y_pred)
                return r2
            elif index == 'rmse':
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                return rmse
            else:
                raise Exception('Wrong index selection.')
        elif self.args['prob'] == 'classification':
            acc = accuracy_score(y, y_pred)
            return acc
        else:
            raise Exception('Wrong problem type.')
