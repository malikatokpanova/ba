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
import wandb
import networkx as nx


config=dict(
        lr=0.001,
        seed=0,
        batch_size=3,
        epochs=1000,
)

graph_parameters={
    'num_nodes': 5,   
    'clique_r':3,
    'clique_s':3,
    'num_classes':2
}


num_nodes=graph_parameters['num_nodes']
clique_r=graph_parameters['clique_r']
clique_s=graph_parameters['clique_s']
num_classes=graph_parameters['num_classes']

num_edges=num_nodes*(num_nodes-1)//2
input_dim=2




def loss_func(probs, cliques_r, cliques_s,num_classes=2):
    loss = 0
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    #dict to map edge to index
    edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))} 
    edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()}) #add reverse edges
    
    for clique in cliques_r:
        #probabilities for an edge being blue
        probs_blue=probs[:,1]
        #get all edges in the clique
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()] #get indices of edges in the clique
        edge_probs = probs_blue[edge_indices] #get probabilities of edges in the clique
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

def cost_soft(probs_blue, cliques_r, cliques_s,num_classes=2):
    loss = 0
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    #dict to map edge to index
    edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))} 
    edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()}) #add reverse edges
    
    for clique in cliques_r:
        #get all edges in the clique
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()] #get indices of edges in the clique
        edge_probs = probs_blue[edge_indices] #get probabilities of edges in the clique
        blue_prod = edge_probs.prod()
        loss += blue_prod
        
    for clique in cliques_s:
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        edge_probs = probs_blue[edge_indices]
        red_prod = (1-edge_probs).prod()
        loss += red_prod
    
    N = cliques_r.size(0) + cliques_s.size(0)
    
    return loss / N

def train_model(x, optimizer, all_cliques_r, all_cliques_s,batch_size,num_nodes,num_epochs):
    #num_epochs = 10000
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.001)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        cliques_r = random.sample(all_cliques_r.tolist(),batch_size)
        cliques_r=torch.tensor(cliques_r,dtype=torch.long)
        cliques_s = random.sample(all_cliques_s.tolist(),batch_size)
        cliques_s=torch.tensor(cliques_s,dtype=torch.long)
    
        probs=F.softmax(x, dim=1)
        loss = loss_func(probs, cliques_r, cliques_s,num_classes)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            wandb.log({'epoch': epoch, 'loss': loss.item()})
        if epoch==num_epochs-1:
            prob_matrix=wandb.Table(data=probs.tolist(),columns=["red","blue"])
           
            wandb.log({"probs":prob_matrix})

    return probs

def cost_func(edge_classes, edge_dict, cliques_r, cliques_s):
    cost = 0
    
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
    
    return cost
    

def evaluate(x, cliques_r, cliques_s):
    with torch.no_grad():
        probs=F.softmax(x, dim=1)
        edge_list = torch.combinations(torch.arange(num_nodes), 2)
        # dict to map edge to index
        edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))}
        edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()})

        sets_thr = torch.argmax(probs, dim=1)
        thresholded_cost = cost_func(sets_thr,edge_dict, cliques_r, cliques_s)
        sets, cost= decode_graph(probs, edge_dict, cliques_r, cliques_s)
        wandb.log({'thresholded_cost':thresholded_cost, 'cost':cost})  
    return cost, sets

#retrieve the solution with the method of conditional expectation
def decode_graph(probs, edge_dict, cliques_r, cliques_s):
    class_probs=probs[:,1]
    sorted_id = torch.argsort(class_probs, descending=True)
    sets = class_probs.detach().clone()

    for idx in sorted_id:
        graph_probs_0 = sets.clone()
        graph_probs_1 = sets.clone()
        
        graph_probs_0[idx] = 0  # Edge is red
        graph_probs_1[idx] = 1  # Edge is blue
        
        
        expected_obj_0 = cost_soft(graph_probs_0, cliques_r, cliques_s)  
        expected_obj_1 = cost_soft(graph_probs_1, cliques_r, cliques_s)  
        
        if expected_obj_0 > expected_obj_1:
            sets[idx] = 1  
            
        else:
            sets[idx] = 0 

    
    expected_obj_G = cost_func(sets, edge_dict, cliques_r, cliques_s)
    return sets, expected_obj_G  # Return the coloring and its cost
        
    



def make_config(config):
    cliques_r=torch.combinations(torch.arange(num_nodes),clique_r)
    cliques_s=torch.combinations(torch.arange(num_nodes),clique_s)
    #x=torch.randn(num_edges,input_dim, requires_grad=True) #normal distribution
    x=torch.rand(num_edges,num_classes, requires_grad=True) # uniform distribution
    
    optimizer= torch.optim.Adam([x], lr=config.lr)
    return optimizer, cliques_r, cliques_s, x

def model_pipeline(hyperparameters):
    with wandb.init(project="project", config=hyperparameters):
        config = wandb.config
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        optimizer, cliques_r, cliques_s,x = make_config(config)
        train_model(x, optimizer, cliques_r, cliques_s, config.batch_size,num_nodes, config.epochs)
        torch.save(x,f'baseline_{num_nodes}_{config.seed}_{config.batch_size}_{config.lr}_{config.epochs}.pth')
        x=torch.load(f'baseline_{num_nodes}_{config.seed}_{config.batch_size}_{config.lr}_{config.epochs}.pth')
        cost, sets = evaluate(x, cliques_r, cliques_s)
        print(cost, sets)
        return 
    
model_pipeline(config)


