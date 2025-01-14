
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

class ramsey_MPNN(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels,num_features,num_layers,conv_type, num_classes=2):
        super(ramsey_MPNN, self).__init__()
        self.num_features=num_features
        self.num_nodes=num_nodes
        self.hidden_channels=hidden_channels
        self.momentum = 0.1
        self.numlayers=num_layers
        
        # initialize node features as learnable parameters with values from uniform distribution
        self.node_features = torch.nn.Parameter(torch.rand(num_nodes, num_features),requires_grad=True) 
        self.num_classes=num_classes
        self.conv_type=conv_type
        
        self.convs=nn.ModuleList()
        self.bns = nn.ModuleList()
        #  gnn layers based on the number of layers and the type of gnn
        if num_layers > 1:
            for i in range(num_layers - 2):
                if self.conv_type == 'GIN':
                    self.convs.append(GINConv(Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                    ), train_eps=True))
                elif self.conv_type == 'GCN':
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            if self.conv_type == 'GIN':
                self.conv1 = GINConv(Sequential(Linear(num_features, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                ), train_eps=True)
                self.convlast = GINConv(Sequential(Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, num_features),
                    ReLU(),
                ), train_eps=True)
            elif self.conv_type == 'GCN':
                self.conv1 = GCNConv(num_features, hidden_channels)
                self.convlast = GCNConv(hidden_channels, num_features)
                
        # if we have only one GNN layer
        else:
            if self.conv_type == 'GIN':
                self.conv1 = GINConv(Sequential(Linear(num_features,  hidden_channels),
                    ReLU(),
                    Linear( hidden_channels,  num_features),
                    ReLU(),
                ),train_eps=True) 
                self.convlast = self.conv1
            elif self.conv_type == 'GCN':
                self.conv1 = GCNConv(num_features, hidden_channels)
                self.convlast = GCNConv(hidden_channels, num_features)
        # initialize edge prediction network    
        self.edge_pred_net = EdgePredNet(num_features,hidden_channels,num_classes) 
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv, bn in zip(self.convs, self.bns):
            conv.reset_parameters()
            bn.reset_parameters()
                
        self.convlast.reset_parameters()
        
        self.edge_pred_net.bn5.reset_parameters()
        nn.init.xavier_uniform_(self.edge_pred_net.lin5.weight)
        nn.init.zeros_(self.edge_pred_net.lin5.bias)
        
        nn.init.xavier_uniform_(self.edge_pred_net.lin6.weight)
        nn.init.zeros_(self.edge_pred_net.lin6.bias)
        
    def forward(self,x):
        # input x is the placeholder, in practice we initialize node features above
        # as a learnable parameter from the uniform distribution
        x = self.node_features.to(x.device)
        
        num_nodes = x.shape[0]
        
        # generate all possible pairs of nodes for a given number of nodes (i.e. all edges for a complete graph structure)
        # e.g. num_nodes=3, then output is tensor([[0, 0, 1], [1, 2, 2]])
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).t()
        
        xinit=x.clone()
        if self.numlayers > 1:
            # apply the first gnn layer
            x=F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
            # iterate over the remaining layers
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = F.leaky_relu(x, negative_slope=0.01)
                x = bn(x)
            
            x=self.convlast(x, edge_index)
        else:
            x=F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
            
        x=x+xinit  #skip connection
         
        
        
        edge_pred = self.edge_pred_net(x, edge_index)
       
        
        # create empty tensor to store probabilities
        probs = torch.zeros(num_nodes, num_nodes, self.num_classes, device=x.device)
        # for each edge, assign the predicted probabilities to both directions of the edge
        # probs stores probabilities of each edge being in each class for the entire graph.
        # each entry probs[i,j] represents the probabilities of the edge between node i and node j being of certain color
        # i.e. probs[i,j,0] is the probability of and edge (i,j) being of color blue
        probs[edge_index[0], edge_index[1]] = edge_pred 
        probs[edge_index[1], edge_index[0]] = edge_pred
        return probs
    

class EdgePredNet(torch.nn.Module):
    def __init__(self,num_features,hidden_channels, num_classes):
        super(EdgePredNet, self).__init__() 
        self.lin5=Linear(num_features,hidden_channels) 
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.lin6=Linear(hidden_channels,num_classes)
    def forward(self, x, edge_index):
        # extract source node features using edge indices
        x_i = x[edge_index[0], :] #edge_index[0] contains the source nodes
        # extract target node features
        x_j = x[edge_index[1], :] #edge_index[1] contains the target nodes
        edge_pred= F.leaky_relu(self.lin5(x_i * x_j), negative_slope=0.01) #elementwise mult
        edge_pred=self.bn5(edge_pred) 
        edge_pred=self.lin6(edge_pred) 
        # apply softmax to obtain a probability distribution over the colors for each edge, 
        # i.e. probs is a tensor where each row (each row represents an edge) sums to 1 
        # each column represents a probability belonging to a certain color
        edge_pred = F.softmax(edge_pred, dim=-1)
        return edge_pred


# loss function 
# input probabilities tensor from above
# batch of cliques of size r -> cliques_r
# batch of cliques of size s -> cliques_s
def loss_func(probs, cliques_r,cliques_s):
    device = probs.device
    loss = 0
    cliques_r = cliques_r.to(device)
    cliques_s = cliques_s.to(device)
    
    # iterate over each clique of size r 
    for clique in cliques_r:   
        # generate all possible edges for a given clique 
        edge_indices=torch.combinations(clique, r=2).t()
        # we consider only edges where the first node index is less 
        # than the second (to not consider duplicate edges and diagonal)
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        # extract the probabilities of the edges being blue 
        edge_probs = probs[edge_indices[0], edge_indices[1],0]
        
        # compute the product of the blue edge probabilities and add to the loss
        blue_prod = edge_probs.prod()
        
        loss += blue_prod 
    # same as above but for cliques of size s and color red( i.e. probability[edge is red]=1 - probability[edge is blue])
    for clique in cliques_s:
        edge_indices=torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs[edge_indices[0], edge_indices[1],0]
        # alternatively(if we have more colors: edge_probs = probs[edge_indices[0], edge_indices[1],1] and red_prod = edge_probs.prod())

        red_prod = (1 - edge_probs).prod()
    
        loss += red_prod 
        
    N = cliques_r.size(0) + cliques_s.size(0)
    
        
    return loss/N

    
# cost function same as loss but instead we use it to count monochromatic cliques
def cost(probs, cliques_r,cliques_s):
    device = probs.device
    expectation = 0
    cliques_r = cliques_r.to(device)
    cliques_s = cliques_s.to(device)
    
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