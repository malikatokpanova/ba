
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU, BatchNorm1d, Dropout

from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv, GatedGraphConv

from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

class ramsey_MPNN(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels,num_features,num_layers,dropout,num_classes=2):
        super(ramsey_MPNN, self).__init__()
        self.num_features=num_features
        self.num_nodes=num_nodes
        self.hidden_channels=hidden_channels
        self.momentum = 0.1
        #self.node_embedding = nn.Embedding(num_nodes, num_features)
        self.numlayers=num_layers
        self.dropout=dropout
        self.node_features = torch.nn.Parameter(torch.rand(num_nodes, num_features),requires_grad=True) 
        self.num_classes=num_classes
        
        
        self.convs=nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                ), train_eps=True)) 
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
        self.conv1 = GINConv(Sequential(Linear(num_features,  hidden_channels),
            ReLU(),
            Linear( hidden_channels,  hidden_channels),
            ReLU(),
        ),train_eps=True) 
        
            
        self.lin1=Linear(hidden_channels,hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.lin2=Linear(hidden_channels,hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin3=Linear(hidden_channels,num_features)
        self.edge_pred_net = EdgePredNet(num_features,hidden_channels,num_classes,dropout) 
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv, bn in zip(self.convs, self.bns):
            conv.reset_parameters()
            bn.reset_parameters()
                
        
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        nn.init.xavier_uniform_(self.lin3.weight)
        nn.init.zeros_(self.lin3.bias)
        
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.edge_pred_net.bn5.reset_parameters()

        nn.init.xavier_uniform_(self.edge_pred_net.lin5.weight)
        nn.init.zeros_(self.edge_pred_net.lin5.bias)
        
        nn.init.xavier_uniform_(self.edge_pred_net.lin6.weight)
        nn.init.zeros_(self.edge_pred_net.lin6.bias)
        
    def forward(self,x):
        x = self.node_features.to(x.device)
        #x=self.node_embedding.weight
        num_nodes = x.shape[0]
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).t()
        
        xinit=x.clone()
         
        x=F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
        for conv, bn in zip(self.convs, self.bns):
            x = F.leaky_relu(conv(x, edge_index), negative_slope=0.01)
            x = bn(x)
            
    
         
        
        x=F.leaky_relu(self.lin1(x),negative_slope=0.01)
        x=self.bn1(x)
        #x=F.dropout(x, p=self.dropout, training=self.training) 
        x=F.leaky_relu(self.lin2(x),negative_slope=0.01) 
        x=self.bn2(x)
        #x=F.dropout(x, p=self.dropout, training=self.training)
        x=self.lin3(x)
        x=x+xinit  #skip connection  
                  
        
        
        edge_pred = self.edge_pred_net(x, edge_index, xinit)
        edge_pred = F.softmax(edge_pred, dim=-1)
        
        probs = torch.zeros(num_nodes, num_nodes, self.num_classes, device=x.device)
        probs[edge_index[0], edge_index[1]] = edge_pred 
        probs[edge_index[1], edge_index[0]] = edge_pred
        return probs
    

class EdgePredNet(torch.nn.Module):
    def __init__(self,num_features,hidden_channels, num_classes, dropout):
        super(EdgePredNet, self).__init__() 
        #self.lin5=Linear(num_features,hidden_channels) #elementwise mult
        self.lin5=Linear(2*num_features,hidden_channels) #concat
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.lin6=Linear(hidden_channels,num_classes)
    def forward(self, x, edge_index, xinit):
        x_i = x[edge_index[0], :] #edge_index[0] contains the source nodes
        x_j = x[edge_index[1], :] #edge_index[1] contains the target nodes
        edge_features = torch.cat([x_i, x_j], dim=-1)  #concat

        edge_pred= F.leaky_relu(self.lin5(edge_features), negative_slope=0.01) # concat
        #edge_pred= F.leaky_relu(self.lin5(x_i * x_j), negative_slope=0.01) #elementwise mult
        edge_pred=self.bn5(edge_pred) #followed by
        edge_pred=self.lin6(edge_pred) #followed by 
        return edge_pred

def loss_func(probs, cliques_r,cliques_s):
    loss = 0
    
    for clique in cliques_r:   
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1],0]
        
        blue_prod = edge_probs.prod()
        
        loss += blue_prod 
    for clique in cliques_s:
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1],0]

        red_prod = (1 - edge_probs).prod()
    
        loss += red_prod 
    
    """ if len(cliques_r[0])!=len(cliques_s[0]):
        N=len(cliques_r)+len(cliques_s)
    else:
        N=len(cliques_r) """
    
        
    N = cliques_r.size(0) + cliques_s.size(0)
    
        
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



""" def loss_func(probs, cliques_r,cliques_s):
    loss = 0
    
    for clique in cliques_r:   
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1],0]
        
        blue_prod = edge_probs.prod()
        
        loss += blue_prod 
    for clique in cliques_s:
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1],0]

        red_prod = (1 - edge_probs).prod()
    
        loss += red_prod 
    
        
    N = cliques_r.size(0) + cliques_s.size(0)
    
        
    return loss/N
"""
