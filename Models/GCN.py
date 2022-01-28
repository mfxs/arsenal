# GCN: Graph Convolutional Networks
from Base.utils import *


# Network
class GraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(1024,), scale=0.4, prob='regression'):
        super(GraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net = [dim_X, ] + list(gc)
        self.scale = scale
        self.prob = prob

        # Model creation
        self.gc = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()
        if prob == 'regression':
            for i in range(dim_y):
                self.gc.append(nn.ModuleList())
                for j in range(len(gc)):
                    self.gc[-1].append(GraphConvolution(self.net[j], self.net[j + 1]))
                self.fc.append(nn.Linear(gc[-1], 1))
        elif prob == 'classification':
            for i in range(len(gc)):
                self.gc.append(GraphConvolution(self.net[i], self.net[i + 1]))
            self.fc = nn.Linear(gc[-1], dim_y)
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        adj = adjacency_matrix(X.cpu().numpy(), 'rbf', scale=self.scale, gpu=X.device)
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                feat = X
                for j in self.gc[i]:
                    feat = self.act(j(feat, adj))
                feat = self.fc[i](feat)
                res_list.append(feat.squeeze())
            res = torch.stack(res_list, dim=-1)
        elif self.prob == 'classification':
            res = X
            for i in self.gc:
                res = i(res, adj)
            res = self.fc(res)
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class GcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, **args):
        super(GcnModel, self).__init__()

        # Parameter assignment
        self.args['gc'] = (1024,)
        self.args['scale'] = 0.4
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Train
    def fit(self, X, y):
        # Data creation
        self.data_create(X, y)

        # Model creation
        self.model = GraphConvolutionalNetworks(self.dim_X, self.dim_y, self.args['gc'], self.args['scale'],
                                                self.args['prob']).cuda(self.args['gpu'])
        self.model_create()

        # Model training
        self.training()

        return self
