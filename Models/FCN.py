# FCN: Fully Connected Networks
from Base.utils import *
from sklearn.neural_network import MLPRegressor, MLPClassifier


# Network
class FullyConnectedNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, hidden_layers=(256,), prob='regression'):
        super(FullyConnectedNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.network_structure = [dim_X, ] + list(hidden_layers)
        self.prob = prob

        # Model creation
        self.net = nn.ModuleList()
        if prob == 'regression':
            for i in range(dim_y):
                self.net.append(nn.ModuleList())
                for j in range(len(hidden_layers)):
                    self.net[-1].append(
                        nn.Sequential(nn.Linear(self.network_structure[j], self.network_structure[j + 1]), nn.ReLU()))
                self.net[-1].append(nn.Linear(self.network_structure[-1], 1))
        elif prob == 'classification':
            for i in range(len(hidden_layers)):
                self.net.append(
                    nn.Sequential(nn.Linear(self.network_structure[i], self.network_structure[i + 1]), nn.ReLU()))
            self.net.append(nn.Linear(self.network_structure[-1], dim_y))
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                feat = X
                for j in self.net[i]:
                    feat = j(feat)
                res_list.append(feat.squeeze())

            res = torch.stack(res_list, dim=1)
        elif self.prob == 'classification':
            res = X
            for i in self.net:
                res = i(res)
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class FcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, **args):
        super(FcnModel, self).__init__()

        # Parameter assignment
        self.args['hidden_layers'] = (256,)
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Train
    def fit(self, X, y):
        # Data creation
        self.data_create(X, y)

        # Model creation
        self.model = FullyConnectedNetworks(self.dim_X, self.dim_y, self.args['hidden_layers'], self.args['prob']).cuda(
            self.args['gpu'])
        self.model_create()

        # Model training
        self.training()

        return self
