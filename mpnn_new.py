#import numpy as np
#import matplotlib.pyplot as plt
from itertools import combinations, chain, islice
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
from new_model import ramsey_MPNN, loss_func, cost

import wandb


import argparse
from pathlib import Path

""" parser = argparse.ArgumentParser()

#parser.add_argument('num_nodes', type=int, help='Number of nodes')
parser.add_argument('--hidden_channels', type=int, help='Number of hidden channels', default=32)
parser.add_argument('--num_features', type=int, help='Number of features', default=32)

parser.add_argument('--learning_rate_1', type=float, help='Learning rate 1', default=0.001)
parser.add_argument('--learning_rate_2', type=float, help='Learning rate 2', default=0.01)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)

args = parser.parse_args()

#num_samples = args.num_samples
hidden_channels=args.hidden_channels
num_features=args.num_features
lr_1=args.learning_rate_1
lr_2=args.learning_rate_2

epochs = args.epochs """

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config=dict(
        hidden_channels= 64,
        num_features= 32,
        lr_1=0.001,
        lr_2=0.01,
        seed=0,
        num_layers=5,
        epochs=6000,
)

graph_parameters={
    'num_nodes': 13,   
    'num_cliques':64, 
    'clique_r':3,
    'clique_s':5
}


num_nodes=graph_parameters['num_nodes']
num_cliques=graph_parameters['num_cliques']
clique_r=graph_parameters['clique_r']
clique_s=graph_parameters['clique_s']


lr_decay_step_size = 20
lr_decay_factor = 0.95


retdict = {}
""" edge_drop_p = 0.0
edge_dropout_decay = 0.90
 """

#for plotting loss values
train_loss_dict={}
threshold=0.0005

#train
def train_model(net,optimizer_1,optimizer_2,num_nodes, hidden_channels,num_features, learning_rate_1,learning_rate_2, epochs, lr_decay_step_size, lr_decay_factor, clique_r, num_cliques,all_cliques_r,all_cliques_s):
    
    net.train()
    wandb.watch(net,log='all',log_freq=10)
    
    for epoch in range(epochs):
        count=0
        if epoch == 4000:
            net.node_features.requires_grad = False 
        """ if epoch % 5 == 0:
            edge_drop_p = edge_drop_p*edge_dropout_decay
            print("Edge_dropout: ", edge_drop_p) """

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
                    
        count += 1 
        
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
            
        #cliques_r=torch.randint(0,num_nodes, (num_cliques, clique_r))
        #cliques_s=torch.randint(0,num_nodes, (num_cliques, clique_s))
        cliques_r=random.sample(all_cliques_r.tolist(),num_cliques)
        cliques_s=random.sample(all_cliques_s.tolist(),num_cliques)
        
        cliques_r=torch.tensor(cliques_r,dtype=torch.long).to(device)
        cliques_s=torch.tensor(cliques_s,dtype=torch.long).to(device)
        #cliques_r=all_cliques_r
        #cliques_s=all_cliques_s
        #cliques=torch.combinations(torch.arange(num_nodes),clique_r)
        probs=net(torch.randn(net.num_nodes, net.num_features).to(device))
        loss=loss_func(probs,cliques_r,cliques_s)
        loss.backward()
        
        if epoch%10==0 or epoch==epochs-1:
            wandb.log({"epoch": epoch, "loss": loss.item()})
            #print('Epoch: ', epoch, 'loss:', loss.item())
            #variance=net.node_embedding.weight.detach().cpu().var(dim=0).mean().item()
            variance=net.node_features.detach().cpu().var(dim=0).mean().item()
            wandb.log({"epoch": epoch, "node embeddings variance": variance})
            
            if variance<threshold:
                print(f"Early stopping at epoch {epoch}, low variance: {variance}")
                break
            if variance < 0.001:
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] *= 0.9
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] *= 0.9
                    
        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
        
        optimizer_1.step()
        optimizer_2.step()

        train_loss_dict[epoch]=loss.item() 
        
        

#torch.manual_seed(0)

def make(config):
    net=ramsey_MPNN(num_nodes, config.hidden_channels,config.num_features, config.num_layers).to(device) 
    net.to(device).reset_parameters()
    params=[param for name, param in net.named_parameters() if 'edge_pred_net' not in name]
    optimizer_1=Adam(params, lr=config.lr_1, weight_decay=0.0)
    optimizer_2= Adam(net.edge_pred_net.parameters(), lr=config.lr_2, weight_decay=0.0)

    all_cliques_r=torch.combinations(torch.arange(num_nodes),clique_r).to(device) 
    all_cliques_s=torch.combinations(torch.arange(num_nodes),clique_s).to(device)
    #train_model(net,optimizer_1,optimizer_2,num_nodes,config.hidden_channels,config.num_features,config.lr_1, config.lr_2,  config.epochs, lr_decay_step_size, lr_decay_factor, clique_r, num_cliques,all_cliques_r,all_cliques_s)#,hidden_2,edge_drop_p,edge_dropout_decay)
    #torch.save(net.state_dict(), f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}.pth')
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
""" def mc_sampling_new(probs, num_samples):
    samples = torch.zeros(num_samples, *probs.shape)
    
    for i in range(num_samples):
        samples[i] = torch.bernoulli(probs)
    return samples

def optimal_new(all_cliques_r,all_cliques_s,num_samples, samples): 
    #samples=mc_sampling(probs,num_samples)
    costs=torch.zeros(num_samples)
    for i in range(num_samples):
        cost_p=cost(samples[i], all_cliques_r,all_cliques_s)
        costs.scatter_add_(0,torch.tensor([i]),cost_p)
    min_cost, min_index = torch.min(costs,0)
    return min_cost #samples[min_index] """

#retrieve deterministically
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
        expected_obj_0 = loss_func(graph_probs_0, cliques_r,cliques_s) #initial, edge is red
        expected_obj_1 = loss_func(graph_probs_1, cliques_r,cliques_s)
            
        if expected_obj_0 > expected_obj_1: 
            sets[src, dst] = 1  # Edge is blue
            sets[dst,src] = 1  
        else:
            sets[src, dst] = 0  # Edge is red
            sets[dst,src] = 0  
    expected_obj_G = cost(sets, cliques_r,cliques_s)
    return sets, expected_obj_G.detach() #returning the coloring and its cost

    
def evaluate(net,cliques_r,cliques_s, hidden_channels,num_features,lr_1,lr_2,seed,num_layers):
    results = {}
    results_sampling={}
    num_samples=100
    #net=ramsey_MPNN(num_nodes, hidden_channels,num_features) 
    with torch.no_grad():
        
        #net.load_state_dict(torch.load(f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}.pth'))
        #net.load_state_dict(torch.load(f'model_{num_nodes}_{hidden_channels}_{lr_1}_{lr_2}.pth'))
        net.eval()
        probs=net(torch.randn(net.num_nodes, net.num_features).to(device))
        results_fin=decode_graph(num_nodes,probs,cliques_r,cliques_s)
        """ coloring=mc_sampling_new(probs, num_samples)
        results_sampling[num_nodes]=optimal_new(cliques_r,cliques_s,num_samples, coloring) """
        
        """  params_key = str(params)
        if params_key not in results:
            results[params_key] = {}
            
        results[params_key][num_nodes]=results_fin
        """
        wandb.log({"cost": results_fin[1]})#, "coloring":results_fin[0]})
    torch.onnx.export(net, torch.randn(net.num_nodes, net.num_features), f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}_{seed}_{num_layers}.onnx')
    wandb.save(f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}_{seed}_{num_layers}.onnx')
    return results_fin
        
#results, sets=evaluate(num_nodes, clique_r, clique_s)

def model_pipeline(hyperparameters):
    with wandb.init(project="project", config=hyperparameters):
        config = wandb.config
        
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        net, optimizer_1, optimizer_2, all_cliques_r, all_cliques_s = make(config)
        net.to(device)
        train_model(net,optimizer_1,optimizer_2,num_nodes,config.hidden_channels,config.num_features,config.lr_1, config.lr_2,  config.epochs, lr_decay_step_size, lr_decay_factor, clique_r, num_cliques,all_cliques_r,all_cliques_s)#,hidden_2,edge_drop_p,edge_dropout_decay)
        
        torch.save(net.state_dict(), f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}.pth')
        net.load_state_dict(torch.load(f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}.pth'))
        
        evaluate(net,all_cliques_r,all_cliques_s,config.hidden_channels,config.num_features,config.lr_1,config.lr_2, config.seed,config.num_layers)
        #print(evaluate(net,all_cliques_r,all_cliques_s,config.hidden_channels,config.num_features,config.lr_1,config.lr_2))
        
        return net
    
#wandb.agent(sweep_id, model_pipeline)
net=model_pipeline(config)


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