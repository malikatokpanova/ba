import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU, BatchNorm1d, Dropout

from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv, GatedGraphConv

from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

# edge probs contains tensors of dimension (num_edges_clique, num_classes),
# where each tensor contains the probabilities for each edge in the clique in the batch


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
        self.node_features = torch.nn.Parameter(torch.randn(num_nodes, num_features),requires_grad=True) 
        self.num_classes=num_classes
        
        self.convs=nn.ModuleList()
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    BN(hidden_channels, momentum=self.momentum),
                ), train_eps=True))
                
        
        self.conv1 = GINConv(Sequential(Linear(num_features,  hidden_channels),
            ReLU(),
            Linear( hidden_channels,  hidden_channels),
            ReLU(),
            BN(hidden_channels, momentum=self.momentum),
        ),train_eps=True)
        
        #self.node_features = torch.nn.Parameter(torch.randn(num_nodes, num_features),requires_grad=True) 
        #self.node_features = torch.nn.Parameter(torch.empty(num_nodes, num_features))
        
        self.edge_pred_net = EdgePredNet(num_features,hidden_channels,num_classes, dropout) 
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters() 
        
        self.edge_pred_net.lin5.reset_parameters()
        self.edge_pred_net.lin6.reset_parameters()
        
    def forward(self,x, cliques_r, cliques_s):
        x = self.node_features
        #x=self.node_embedding.weight
        num_nodes = x.shape[0]
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).t()
        
        # for each batch, filter the nodes in the clique
        cliques_r_embed=x[cliques_r]
        # get all edges in the clique
        #edge_r=torch.combinations(cliques_r, r=2).t()
        #same for a batch of cliques of size s
        cliques_s_embed=x[cliques_s]
        #edge_s=torch.combinations(cliques_s, r=2).t()
        
                  
        edge_probs_r = []
        for clique,node in zip(cliques_r_embed,cliques_r):
            print(clique, "clique embed")
            print(node, "nodes of a clique")
            idx=torch.combinations(node, r=2).t()
            # passing the clique embedding and the edge indices to the edge prediction network
            print(F.softmax(self.edge_pred_net(clique, idx, node), dim=-1), "edge_prob_tensor")
            
            break
            #edge_probs_r.append(edge_prob_tensor)

        edge_probs_s = []
        """  for clique,idx in zip(cliques_s_embed,edge_s):
            edge_probs_s.append(F.softmax(self.edge_pred_net(clique,idx), dim=-1)) """
        for clique, node in zip(cliques_s_embed, cliques_s):
            idx=torch.combinations(node, r=2).t()
            # passing the clique embedding and the edge indices to the edge prediction network
            edge_probs_s_list=[F.softmax(self.edge_pred_net(clique,idx,node), dim=-1)]
            edge_prob_tensor = torch.stack(edge_probs_s_list)
            edge_probs_s.append(edge_prob_tensor)

        return torch.stack(edge_probs_r), torch.stack(edge_probs_s) #probabilities for all the edges in the batch of cliques



class EdgePredNet(torch.nn.Module):
    def __init__(self,num_features,hidden_channels, num_classes, dropout):
        super(EdgePredNet, self).__init__() 
        #self.lin = Sequential(Linear(2*num_features, hidden_channels), ReLU(), Linear(hidden_channels, 1),torch.nn.Sigmoid())
        #self.dropout=dropout
        """ self.lin1=Linear(num_features,hidden_channels)
        self.lin2=Linear(hidden_channels,hidden_channels)
        self.lin3=Linear(hidden_channels,hidden_channels)
        self.lin4=Linear(hidden_channels,num_features) """
        self.lin5 = Linear(num_features, hidden_channels)
        self.lin6 = Linear(hidden_channels, num_classes)
    def forward(self, x, idx,node):
        """ xinit=x.clone()
        x=F.leaky_relu(self.lin1(x))
        x=F.dropout(x, p=self.dropout, training=self.training) 
        x=F.leaky_relu(self.lin2(x)) 
        x=F.dropout(x, p=self.dropout, training=self.training)
        #x=F.leaky_relu(self.lin3(x))
        #x=F.dropout(x, p=self.dropout, training=self.training)
        x=self.lin4(x)
        x=x+xinit #skip connection """
        index_mapping = {original_idx.item(): pos for pos, original_idx in enumerate(node)}
        edge_index = torch.tensor([index_mapping[idx.item()] for idx in node])
        print(edge_index, "edge_index")
        x_i = x[edge_index[0]] #edge_index[0] contains the source nodes
        print(x_i, "x_i")
        x_j = x[edge_index[1]] #edge_index[1] contains the target nodes
        #edge_features = torch.cat([x_i, x_j], dim=-1)  
        edge_pred=self.lin6(F.relu(self.lin5(x_i * x_j))) 
        return edge_pred

def loss_func(probs_r,probs_s,cliques_r,cliques_s):
    loss = 0
    print(probs_r.size())
    print(probs_r, "probs_r")
    # batch of cliques of size r
    for i, clique in enumerate(cliques_r):
        edge_indices = torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        
        edge_probs = probs_r[i][:,0] #selecting probabilities for blue
        
        blue_prod = edge_probs.prod()
        
        loss += blue_prod.sum() 
        
    for i,clique in enumerate(cliques_s):
        edge_indices = torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs_s[i][:,0] #selecting probabilities for blue

        red_prod = (1 - edge_probs).prod()
        
        #alternatively 
        # edge_probs=probs_s[i][:,1] for red
        # and red_prod=edge_probs.prod() 
        
        loss += red_prod 
    
    
        
    N = cliques_r.size(0) + cliques_s.size(0)
    
        
    return loss/N

# need to adjust
#evaluation
def cost(probs_r,probs_s, cliques_r,cliques_s):
    expectation = 0
    
    for i, clique in enumerate(cliques_r):
        edge_indices = torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs_r[:,0] #selecting probabilities for blue
        
        blue_prod = edge_probs.prod()
        
        expectation += blue_prod.sum() 
        
    for i,clique in enumerate(cliques_s):
        edge_indices = torch.combinations(clique, r=2).t()
        edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
        edge_probs = probs_s[:,0] #selecting probabilities for blue

        red_prod = (1 - edge_probs).prod()
        
        #alternatively 
        # edge_probs=probs_s[i][:,1] for red
        # and red_prod=edge_probs.prod() 
        
        expectation += red_prod 
    
    
    
    return expectation