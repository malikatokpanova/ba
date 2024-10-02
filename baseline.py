import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU, BatchNorm1d, Dropout

from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv, GatedGraphConv

from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

import random 

import networkx as nx
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
random.seed(0)

class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_classes):
        super(Net, self).__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3=Linear(hidden_dim,hidden_dim)
        self.fc4=Linear(hidden_dim,hidden_dim)
        self.fc5=Linear(hidden_dim,num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=1)

num_nodes=17
num_edges=num_nodes*(num_nodes-1)//2
input_dim=2
hidden_dim=128
num_classes=2
clique_r=4
clique_s=4
batch_size=256

model = Net(input_dim, hidden_dim, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x=torch.randn(num_edges,input_dim)



def loss_func(probs, cliques_r, cliques_s,num_classes):
    loss = 0
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    #dict to map edge to index
    edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))} 
    edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()})
    
    for clique in cliques_r:
        #probabilities for an edge being blue
        probs_blue=probs[:,1]
        #get all edges in the clique
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        edge_probs = probs_blue[edge_indices]
        blue_prod = edge_probs.prod()
        loss += blue_prod
        
    for clique in cliques_s:
        probs_red=probs[:,0]
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        edge_probs = probs_red[edge_indices]
        red_prod = edge_probs.prod()
        loss += red_prod
    
    N = cliques_r.size(0) + cliques_s.size(0)
    
    return loss / N



def train_model(model,x, optimizer, clique_r, clique_s,batch_size,num_nodes):
    model.train()
    num_epochs = 1000
    all_cliques_r=torch.combinations(torch.arange(num_nodes),clique_r)
    all_cliques_s=torch.combinations(torch.arange(num_nodes),clique_s)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        cliques_r = random.sample(all_cliques_r.tolist(),batch_size)
        cliques_r=torch.tensor(cliques_r,dtype=torch.long)
        cliques_s = random.sample(all_cliques_s.tolist(),batch_size)
        cliques_s=torch.tensor(cliques_s,dtype=torch.long)
    
        probs = model(x)
        
        loss = loss_func(probs, cliques_r, cliques_s,num_classes)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')
    
    return probs

def cost_func(probs, cliques_r, cliques_s):
    cost = 0
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    # dict to map edge to index
    edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))}
    edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()})

    edge_classes = torch.argmax(probs, dim=1)
    for clique in cliques_r:
        clique_edge = torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        # if edges are classified as blue (1)
        blue_edges = edge_classes[edge_indices] == 1
        if blue_edges.all():
            blue_prod = blue_edges.prod()
            cost += blue_prod
        
    for clique in cliques_s:
        clique_edge = torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        # if edges are classified as red (0)
        red_edges = edge_classes[edge_indices] == 0
        if red_edges.all():
            red_prod = red_edges.prod()
            cost += red_prod
    
    return cost, edge_classes
    
#train_model(model, optimizer, clique_r, clique_s, batch_size,num_nodes)    
    
def evaluate(model, x, clique_r, clique_s):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    with torch.no_grad():
        probs = model(x)
        cliques_r=torch.combinations(torch.arange(num_nodes),clique_r)
        cliques_s=torch.combinations(torch.arange(num_nodes),clique_s)
        cost=cost_func(probs, cliques_r, cliques_s)
    return cost

def main():
    probs=train_model(model, x, optimizer,  clique_r, clique_s, batch_size,num_nodes)
    print(probs, 'probs')
    cost , sets= evaluate(model, x, clique_r, clique_s)
    print(f'Final cost: {cost}')
    return sets
main() 


""" def visualize_graph(num_nodes, edge_classes, edge_dict):
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(num_nodes))
    
    # Add edges with colors
    edge_colors = []
    for edge, idx in edge_dict.items():
        G.add_edge(edge[0], edge[1])
        if edge_classes[idx] == 1:
            edge_colors.append('blue')
        else:
            edge_colors.append('red')
    
    # Draw the graph
    pos = nx.circular_layout(G)  # positions for all nodes
    nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_color='lightgray', node_size=500, font_size=10)
    plt.show()


sets=main()

edge_list = torch.combinations(torch.arange(num_nodes), 2)
edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))}
edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()})

visualize_graph(num_nodes, sets, edge_dict)  """