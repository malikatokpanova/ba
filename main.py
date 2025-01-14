import torch
import time

from torch.optim import Adam


import random
from edgenet import ramsey_NN, loss_func, cost

import wandb

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config=dict(
        hidden_channels= 64,
        num_features= 32,
        lr_1=0.001,
        lr_2=0.01,
        seed=0,
        num_layers=5,
        num_cliques=128,
        epochs=7000,
)

graph_parameters={
    'num_nodes': 17,   
    'clique_r':3,
    'clique_s':6,
    'num_classes':2
}


num_nodes=graph_parameters['num_nodes']
clique_r=graph_parameters['clique_r']
clique_s=graph_parameters['clique_s']
num_classes=graph_parameters['num_classes']

lr_decay_step_size = 20
lr_decay_factor = 0.1


def train_model(net,optimizer_1,optimizer_2, epochs, num_cliques,all_cliques_r,all_cliques_s, device=device):
    
    net.train()
    wandb.watch(net,log='all',log_freq=10)
    start_time = time.time()
    
    for epoch in range(epochs):
        # freeze node features
        if epoch == 5000:
            net.node_features.requires_grad = False   
        """ if epoch % lr_decay_step_size == 0
            for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr'] """
                     
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        # sample a random batch of cliques of size r from the list of all cliques of size r
        cliques_r=random.sample(all_cliques_r.tolist(),num_cliques)
        # sample a random batch of cliques of size r from the list of all cliques of size s
        cliques_s=random.sample(all_cliques_s.tolist(),num_cliques)
        # convert to a tensor 
        cliques_r=torch.tensor(cliques_r,dtype=torch.long).to(device)
        cliques_s=torch.tensor(cliques_s,dtype=torch.long).to(device)
        
        # torch.randn(net.num_nodes, net.num_features) is just a placeholder, we initialize the node features in the model initialization 
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
    end_time = time.time()
    wandb.log({'runtime': end_time - start_time})
        
        

def make(config,device):
    net=ramsey_NN(num_nodes, config.hidden_channels,config.num_features, config.num_layers).to(device) 
    net.to(device).reset_parameters()
    # we want to separate the parameters of the edge prediction network and the ramsey_nn to use different learning rates
    params=[param for name, param in net.named_parameters() if 'edge_pred_net' not in name]
    optimizer_1=Adam(params, lr=config.lr_1, weight_decay=0.0)
    optimizer_2= Adam(net.edge_pred_net.parameters(), lr=config.lr_2, weight_decay=0.0)
    
    # given the number of nodes, generate all possible sets of nodes (cliques) of size clique_r
    all_cliques_r=torch.combinations(torch.arange(num_nodes),clique_r).to(device) 
    # given the number of nodes, generate all possible sets of nodes (cliques) of size clique_s
    all_cliques_s=torch.combinations(torch.arange(num_nodes),clique_s).to(device)
   
    return net, optimizer_1, optimizer_2, all_cliques_r, all_cliques_s

# retrieve discrete solutions by applying a threshold 
# each entry probs[i,j] represents the probabilities of the edge between node i and node j being of certain color
# hence probs[i,j,0] is the probability for the edge being blue 
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

#retrieve with the method of conditional expectation
def decode_graph(num_nodes,probs,cliques_r,cliques_s,device):
    # generate all possible pairs of nodes for a given number of nodes (i.e. all edges for a complete graph structure)
    # e.g. num_nodes=3, then output is tensor([[0, 0, 1], [1, 2, 2]])
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(device)
    
    # extract only probabilities for an edge being blue 
    class_probs=probs[:,:,0] 
    class_probs=class_probs.to(device)
    #flat_probs = probs[edge_index[0], edge_index[1]] #if we have (n,n) tensor
    
    # flatten class_probs
    flat_probs = class_probs[edge_index[0], edge_index[1]]
    # sort the edges according to their probabilities in the decreasing order
    sorted_inds = torch.argsort(flat_probs, descending=True)
    
    sets= class_probs.detach().clone().to(device)
    for flat_index in sorted_inds:
        # retrieve the source and target nodes for the current edge
        edge = edge_index[:, flat_index]
        # extract the source node index and the target node index
        src, dst = edge[0].item(), edge[1].item()
        
        # create copies of the probabilties edge being blue to check both scenarios (p_{(i,j)}=1 and p_{(i,j)}=0)
        graph_probs_0 = sets.clone()
        graph_probs_1 = sets.clone()
        
        # set the current edge to red (0) in one copy and blue (1) in another 
        graph_probs_0[src, dst] = 0
        graph_probs_0[dst,src] = 0  
        
        graph_probs_1[src, dst] = 1
        graph_probs_1[dst,src] = 1  
        
        
        # compute the loss if the current edge is red
        expected_obj_0 = cost(graph_probs_0, cliques_r,cliques_s) 
        # compute the loss if the current edge is blue
        expected_obj_1 = cost(graph_probs_1, cliques_r,cliques_s) 
        
        
        # if the condition loss[edge blue]<loss[edge red] is met,
        # i.e. the loss does not increase, set the edge to blue
        # otherwise red
        if expected_obj_0 > expected_obj_1: 
            sets[src, dst] = 1  # edge is blue
            sets[dst,src] = 1  
        else:
            sets[src, dst] = 0  # Edge is red
            sets[dst,src] = 0  
    expected_obj_G = cost(sets, cliques_r,cliques_s)
    return sets, expected_obj_G.detach() #returning the coloring and its cost

    
def evaluate(net,cliques_r,cliques_s, hidden_channels,num_features,lr_1,lr_2,seed,num_layers,num_cliques, epochs,device):

    with torch.no_grad():
        net.eval()
        probs=net(torch.randn(net.num_nodes, net.num_features).to(device))
        results_fin=decode_graph(num_nodes,probs,cliques_r,cliques_s,device)
        
        #-----------IMPORTANCE OF LEARNING EXPERIMENT---------------
        """ uniform_probs=torch.rand(num_nodes,num_nodes,2,device=device)  
        uniform_cost=decode_graph(num_nodes,uniform_probs,cliques_r,cliques_s,device)[1]
        results_fin_thr = discretize(probs, cliques_r,cliques_s)
        wandb.log({"cost": results_fin[1], "thresholded_cost": results_fin_thr[1], 'uniform_cost':uniform_cost}) """
        results_fin_thr = discretize(probs, cliques_r,cliques_s)
        wandb.log({"cost": results_fin[1], "thresholded_cost": results_fin_thr[1]})
    torch.onnx.export(net, torch.randn(net.num_nodes, net.num_features), f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}_{seed}_{num_layers}_{num_cliques}_{epochs}.onnx')
    wandb.save(f'model_{num_nodes}_{hidden_channels}_{num_features}_{lr_1}_{lr_2}_{seed}_{num_layers}_{num_cliques}_{epochs}.onnx')
    return results_fin
        
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
        torch.use_deterministic_algorithms(True)
        
        net, optimizer_1, optimizer_2, all_cliques_r, all_cliques_s = make(config,device)
        net.to(device)
        train_model(net,optimizer_1,optimizer_2,  config.epochs,  config.num_cliques,all_cliques_r,all_cliques_s, device)
        
        torch.save(net.state_dict(), f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}_{config.num_cliques}_{config.epochs}.pth')
        net.load_state_dict(torch.load(f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}_{config.num_cliques}_{config.epochs}.pth'))
        #wandb.save(f'model_{num_nodes}_{config.hidden_channels}_{config.num_features}_{config.lr_1}_{config.lr_2}_{config.seed}_{config.num_layers}_{config.dropout}_{config.num_cliques}.pth')
        evaluate(net,all_cliques_r,all_cliques_s,config.hidden_channels,config.num_features,config.lr_1,config.lr_2, config.seed,config.num_layers,config.num_cliques, config.epochs,device)
        print(evaluate(net,all_cliques_r,all_cliques_s,config.hidden_channels,config.num_features,config.lr_1,config.lr_2, config.seed,config.num_layers,config.num_cliques, config.epochs,device))
        
        return net
    

net=model_pipeline(config)


