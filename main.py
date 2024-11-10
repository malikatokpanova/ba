#import numpy as np
#import matplotlib.pyplot as plt
from itertools import combinations, chain
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
#import networkx as nx
import os

from torch.optim import Adam

from math import ceil

#from matplotlib.pylab import plt
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

import random
from random import sample
from edgenet import ramsey_NN, loss_func, cost

import wandb


import argparse
from pathlib import Path

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config=dict(
        hidden_channels= 64,
        num_features= 32,
        lr_1=0.001,
        lr_2=0.01,
        seed=0,
        num_layers=5,
        dropout=0.3,
        num_cliques=128,
        epochs=10000,
)

graph_parameters={
    'num_nodes': 17,   
    'clique_r':4,
    'clique_s':4,
    'num_classes':2
}


num_nodes=graph_parameters['num_nodes']
clique_r=graph_parameters['clique_r']
clique_s=graph_parameters['clique_s']
num_classes=graph_parameters['num_classes']

lr_decay_step_size = 20
lr_decay_factor = 0.1


retdict = {}


#for plotting loss values
train_loss_dict={}


#train
def train_model(net,optimizer_1,optimizer_2,num_nodes, hidden_channels,num_features, learning_rate_1,learning_rate_2, epochs, lr_decay_step_size, lr_decay_factor, clique_r, num_cliques,all_cliques_r,all_cliques_s, device=device):
    
    net.train()
    wandb.watch(net,log='all',log_freq=10)
    
    for epoch in range(epochs):
        
        """ if epoch == 5000:
            net.node_features.requires_grad = False     """
        """ if epoch % lr_decay_step_size == 0
            for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr'] """
                     
        
        
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        
        cliques_r=random.sample(all_cliques_r.tolist(),num_cliques)
        cliques_s=random.sample(all_cliques_s.tolist(),num_cliques)
        
        cliques_r=torch.tensor(cliques_r,dtype=torch.long).to(device)
        cliques_s=torch.tensor(cliques_s,dtype=torch.long).to(device)
        #cliques_r=all_cliques_r
        #cliques_s=all_cliques_s
        
        
        probs=net(torch.randn(net.num_nodes, net.num_features).to(device))
        loss=loss_func(probs,cliques_r,cliques_s)
        loss.backward()
        
        if epoch%10==0 or epoch==epochs-1:
            wandb.log({"epoch": epoch, "loss": loss.item()})
            
        if epoch==epochs-1:
            node_embeddings = net.node_features.detach().cpu().numpy()
            prob_matrix = probs.detach().cpu().numpy()
            node_embeddings_table = wandb.Table(data=node_embeddings.tolist(), columns=[f"feature_{i}" for i in range(node_embeddings.shape[1])])
            prob_matrix_table = wandb.Table(data=prob_matrix.tolist(), columns=[f"node_{i}" for i in range(prob_matrix.shape[1])])
        
            wandb.log({
                "node_embeddings": node_embeddings_table,
                "prob_matrix": prob_matrix_table
            })

                    
        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
        
        optimizer_1.step()
        optimizer_2.step()

        train_loss_dict[epoch]=loss.item() 
        
        

#torch.manual_seed(0)

def make(config,device):
    net=ramsey_NN(num_nodes, config.hidden_channels,config.num_features, config.num_layers, config.dropout).to(device) 
    net.to(device).reset_parameters()
    params=[param for name, param in net.named_parameters() if 'edge_pred_net' not in name]
    optimizer_1=Adam(params, lr=config.lr_1, weight_decay=0.0)
    optimizer_2= Adam(net.edge_pred_net.parameters(), lr=config.lr_2, weight_decay=0.0)

    all_cliques_r=torch.combinations(torch.arange(num_nodes),clique_r).to(device) 
    all_cliques_s=torch.combinations(torch.arange(num_nodes),clique_s).to(device)
   
    return net, optimizer_1, optimizer_2, all_cliques_r, all_cliques_s

#plotting loss
""" 
plt.clf()
fig,ax=plt.subplots(figsize=(9,7))
epochs_=range(0,epochs)
plt.plot(epochs_,train_loss_dict.values(), label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss')

textstr='\n'.join((
fr'$N={num_nodes}$',
f'$\\beta_1={lr_1}$',
f'$\\beta_2={lr_2}$',
f'$\\gamma={hidden_channels}$',
fr'$\omega={num_features}$'))


props = dict(boxstyle='square', facecolor='white', alpha=0.5)

ax.text(0.82, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


#plt.legend()
#plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.savefig(f'loss_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}.png')
plt.close    
"""   
def discretize(probs, cliques_r,cliques_s,threshold=0.5):
    num_nodes = probs.size(0)
    sets = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=probs.device)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if probs[i, j,0] > threshold:  
                sets[i, j] = 1  # Edge is blue
                sets[j, i] = 1
            else:
                sets[i, j] = 0  # edge is red
                sets[j, i] = 0
    expected_obj_G = cost(sets, cliques_r,cliques_s)
    return sets, expected_obj_G.detach()

#retrieve deterministically
def decode_graph(num_nodes,probs,cliques_r,cliques_s,device):
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(device)
    
    # if we have (n,n,2) tensor
    class_probs=probs[:,:,0] #taking the probabilities of the blue class
    class_probs=class_probs.to(device)
    #flat_probs = probs[edge_index[0], edge_index[1]] if we have (n,n) tensor
    flat_probs = class_probs[edge_index[0], edge_index[1]]
    sorted_inds = torch.argsort(flat_probs, descending=True)
    
    #sets = probs.detach().clone().to(device)
    sets= class_probs.detach().clone().to(device)
    for flat_index in sorted_inds:
        
        edge = edge_index[:, flat_index]
        src, dst = edge[0].item(), edge[1].item()
        
        graph_probs_0 = sets.clone()
        graph_probs_1 = sets.clone()
        
        
        graph_probs_0[src, dst] = 0
        graph_probs_0[dst,src] = 0  
        
        graph_probs_1[src, dst] = 1
        graph_probs_1[dst,src] = 1  
        
        
        expected_obj_0 = cost(graph_probs_0, cliques_r,cliques_s) #initial, edge is red
        expected_obj_1 = cost(graph_probs_1, cliques_r,cliques_s) #edge is blue in the solution
        #expected_obj_0 = loss_func(graph_probs_0, cliques_r,cliques_s) #initially edge is red
        #expected_obj_1 = loss_func(graph_probs_1, cliques_r,cliques_s)
            
        if expected_obj_0 > expected_obj_1: 
            sets[src, dst] = 1  # edge is blue
            sets[dst,src] = 1  
        else:
            sets[src, dst] = 0  # Edge is red
            sets[dst,src] = 0  
    expected_obj_G = cost(sets, cliques_r,cliques_s)
    return sets, expected_obj_G.detach() #returning the coloring and its cost

    
def evaluate(net,cliques_r,cliques_s, hidden_channels,num_features,lr_1,lr_2,seed,num_layers,dropout,num_cliques, epochs,device):

    with torch.no_grad():
        net.eval()
        probs=net(torch.randn(net.num_nodes, net.num_features).to(device))
        results_fin=decode_graph(num_nodes,probs,cliques_r,cliques_s,device)
        """ coloring=mc_sampling_new(probs, num_samples)
        results_sampling[num_nodes]=optimal_new(cliques_r,cliques_s,num_samples, coloring) """
        
        """  params_key = str(params)
        if params_key not in results:
            results[params_key] = {}
            
        results[params_key][num_nodes]=results_fin
        """
        results_fin_thr = discretize(probs, cliques_r,cliques_s)
        wandb.log({"cost": results_fin[1], "thresholded_cost": results_fin_thr[1]})
    torch.onnx.export(net, torch.randn(net.num_nodes, net.num_features), f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}_{seed}_{num_layers}_{dropout}_{num_cliques}_{epochs}.onnx')
    wandb.save(f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}_{seed}_{num_layers}_{dropout}_{num_cliques}_{epochs}.onnx')
    return results_fin
        
#results, sets=evaluate(num_nodes, clique_r, clique_s)

def model_pipeline(hyperparameters):
    with wandb.init(project="project", config=hyperparameters):
        config = wandb.config
        
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        net, optimizer_1, optimizer_2, all_cliques_r, all_cliques_s = make(config,device)
        net.to(device)
        train_model(net,optimizer_1,optimizer_2,num_nodes,config.hidden_channels,config.num_features,config.lr_1, config.lr_2,  config.epochs, lr_decay_step_size, lr_decay_factor, clique_r, config.num_cliques,all_cliques_r,all_cliques_s, device)#,hidden_2,edge_drop_p,edge_dropout_decay)
        
        torch.save(net.state_dict(), f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}_{config.dropout}_{config.num_cliques}_{config.epochs}.pth')
        net.load_state_dict(torch.load(f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}_{config.dropout}_{config.num_cliques}_{config.epochs}.pth'))
        #wandb.save(f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}_{config.dropout}_{config.num_cliques}.pth')
        evaluate(net,all_cliques_r,all_cliques_s,config.hidden_channels,config.num_features,config.lr_1,config.lr_2, config.seed,config.num_layers,config.dropout,config.num_cliques, config.epochs,device)
        print(evaluate(net,all_cliques_r,all_cliques_s,config.hidden_channels,config.num_features,config.lr_1,config.lr_2, config.seed,config.num_layers,config.dropout,config.num_cliques, config.epochs,device))
        
        return net
    

net=model_pipeline(config)


#plot the graph
""" color_dict={0:'red', 1:'blue'}

graph_coloring=nx.Graph()
#sets[0]
for i in range(num_nodes):
    for j in range(num_nodes):
        if i<j:
            if sets[0][i,j]==1:
                graph_coloring.add_edge(i,j, color='blue')
            else:
                graph_coloring.add_edge(i,j, color='red')
            
edge_colors=[graph_coloring[u][v]['color'] for u,v in graph_coloring.edges()]
pos = nx.circular_layout(graph_coloring)  
plt.title('Ramsey (3,3)-graph on 5 vertices')
nx.draw(graph_coloring, pos, edge_color=edge_colors, with_labels=True, node_color='lightgrey', node_size=500)

plt.show()   
"""

"""#retrieve deterministically when we have (n,n) probability tensor
def decode_graph(num_nodes,probs,cliques_r,cliques_s):
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(device)
    flat_probs = probs[edge_index[0], edge_index[1]]
    sorted_inds = torch.argsort(flat_probs, descending=True)
    
    sets = probs.detach().clone().to(device)
    
    for flat_index in sorted_inds:
        
        edge = edge_index[:, flat_index]
        src, dst = edge[0].item(), edge[1].item()
        
        graph_probs_0 = sets.clone()
        graph_probs_1 = sets.clone()
        
        
        graph_probs_0[src, dst] = 0
        graph_probs_0[dst,src] = 0  
        
        graph_probs_1[src, dst] = 1
        graph_probs_1[dst,src] = 1  
        
        
        #expected_obj_0 = cost(graph_probs_0, cliques_r,cliques_s) #initial, edge is red
        #expected_obj_1 = cost(graph_probs_1, cliques_r,cliques_s) #edge is blue in the solution
        expected_obj_0 = loss_func(graph_probs_0, cliques_r,cliques_s) #initially edge is red
        expected_obj_1 = loss_func(graph_probs_1, cliques_r,cliques_s)
            
        if expected_obj_0 > expected_obj_1: 
            sets[src, dst] = 1  # edge is blue
            sets[dst,src] = 1  
        else:
            sets[src, dst] = 0  # Edge is red
            sets[dst,src] = 0  
    expected_obj_G = cost(sets, cliques_r,cliques_s)
    return sets, expected_obj_G.detach() #returning the coloring and its cost
"""