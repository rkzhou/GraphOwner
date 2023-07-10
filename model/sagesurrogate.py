import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv
from utils.graph import numpy_to_graph

# Used for inductive case (graph classification) by default.
class SAGEEMB(nn.Module):  
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu,
                 aggregator_type='mean'):   # mean/gcn/pool/lstm
        super(SAGEEMB, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_dim, hidden_dim[0], aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SAGEConv(hidden_dim[i], hidden_dim[i+1], aggregator_type, feat_drop=dropout, activation=activation))
        
        self.layers.append(SAGEConv(hidden_dim[-1], out_dim, aggregator_type, feat_drop=dropout, activation=activation))
        


    def forward(self, data):
        batch_g = []
        for adj in data[1]:
            # cannot use tensor init DGLGraph
            batch_g.append(numpy_to_graph(adj.detach().cpu().T.numpy(), to_cuda=adj.is_cuda)) 
        batch_g = dgl.batch(batch_g)
        
        B,N,F = data[0].shape[:3]
        x = data[0].reshape(B*N, F)
        for layer in self.layers:
            x = layer(batch_g, x)
        
        embddings = x
        
        return embddings, None