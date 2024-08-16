
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU, BatchNorm1d

from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv, GatedGraphConv

from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


#from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops, dropout_adj


#from layers.mlp_readout_layer import MLPReadout
                    
class ramsey_MPNN(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels,num_features,heads=3):#num_layers, hidden1, hidden2,mask_prob=0.1):
        super(ramsey_MPNN, self).__init__()
        self.num_features=num_features
        self.num_nodes=num_nodes
        self.hidden_channels=hidden_channels
        #self.node_embedding = nn.Embedding(num_nodes, num_features)
        self.node_features = torch.nn.Parameter(torch.randn(num_nodes, num_features),requires_grad=True) 
        
        #torch.nn.init.kaiming_normal_(self.node_features)
        #nn.init.kaiming_uniform_(self.node_features, nonlinearity='relu')
        #nn.init.xavier_uniform_(self.node_features) 
        #nn.init.uniform_(self.node_features, a=0.0, b=1.0)
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden_channels),  BatchNorm1d(hidden_channels),ReLU(), Linear(hidden_channels,hidden_channels), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(), Linear(hidden_channels,num_features), ReLU())) #changed
        #self.conv3 = GINConv(Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(), Linear(hidden_channels, num_features), ReLU()))
        self.lin1=Linear(num_features,hidden_channels)
        
        self.lin2=Linear(hidden_channels,num_features)  
    
        self.edge_pred_net = EdgePredNet(num_features,hidden_channels) 
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        #self.conv3.reset_parameters()
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters() 
        #nn.init.uniform_(self.node_embedding.weight, -0.1, 0.1)
        #torch.nn.init.xavier_uniform_(self.node_features) 
        #torch.nn.init.kaiming_normal_(self.node_features)
        #nn.init.kaiming_uniform_(self.node_features, nonlinearity='relu')
        #nn.init.uniform_(self.node_features, a=0.0, b=1.0)
        #self.edge_pred_net.reset_parameters()    
        #nn.init.normal_(self.node_features, mean=0, std=1) 
    
    def forward(self,x):
        x = self.node_features
        
        #x=self.node_embedding.weight
        num_nodes = x.shape[0]
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).t()
        
        xinit=x
        
        x=self.conv1(x, edge_index) 
        #x=x+xinit
        x=F.leaky_relu(x)
        x=F.dropout(x, p=0.32, training=self.training) 
        x=self.conv2(x, edge_index)
        x=F.leaky_relu(x)
        x=F.dropout(x,p=0.32, training=self.training)
        
        
        """ x=self.conv2(x, edge_index)
        x=F.leaky_relu(x,negative_slope=0.02)
        x=F.dropout(x, p=0.32, training=self.training)
        x=self.conv3(x, edge_index)  """
        #x=x+self.conv2(x, edge_index)
    
    
        x=self.lin1(x)
        x=F.leaky_relu(x)
        x=F.dropout(x, p=0.3, training=self.training) 
        x=self.lin2(x) 
                  
        probs = torch.zeros(num_nodes, num_nodes)
        edge_pred = self.edge_pred_net(x, edge_index)
        
        probs[edge_index[0], edge_index[1]] = edge_pred.squeeze()
        probs[edge_index[1], edge_index[0]] = edge_pred.squeeze() 
        
        
        return probs
    
class EdgePredNet(torch.nn.Module):
    def __init__(self,num_features,hidden_channels):
        super(EdgePredNet, self).__init__() 
        self.lin = Sequential(Linear(2*num_features, hidden_channels), ReLU(), Linear(hidden_channels, 1),torch.nn.Sigmoid())
        #self.lin = Sequential(Linear(2*num_features, hidden_channels), LeakyReLU(), Linear(hidden_channels, 1), torch.nn.Sigmoid())
    def forward(self, x, edge_index):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_features = torch.cat([x_i, x_j], dim=-1)  

        return self.lin(edge_features)

def loss_func(probs, cliques_r,cliques_s):
    loss = 0
    
    for clique in cliques_r:   
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1]]
        
        blue_prod = edge_probs.prod()
        
        loss += blue_prod 
    for clique in cliques_s:
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1]]

        red_prod = (1 - edge_probs).prod()
    
        loss += red_prod 
    
    if len(cliques_r[0])!=len(cliques_s[0]):
        N=len(cliques_r)+len(cliques_s)
    else:
        N=len(cliques_r)
    return loss/N 

    
#evaluation
def cost(probs, cliques_r,cliques_s):
    expectation = 0
    for clique in cliques_r:   
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1]]
        
        blue_prod = edge_probs.prod()
        
        expectation += blue_prod
    for clique in cliques_s:
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1]]

        red_prod = (1 - edge_probs).prod()
    
        expectation += red_prod
    
    return expectation




