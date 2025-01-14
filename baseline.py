import torch
import torch.nn.functional as F
import random 
import wandb
import networkx as nx
import time



config=dict(
        lr=0.001,
        seed=0,
        batch_size=3,
        epochs=10000,
)
# here we consider R(4,4;17) Ramsey graph
graph_parameters={
    'num_nodes': 17,   
    'clique_r':4,
    'clique_s':4,
    'num_classes':2 #number of colors, i.e. for a two-color Ramsey number
}


num_nodes=graph_parameters['num_nodes']
clique_r=graph_parameters['clique_r']
clique_s=graph_parameters['clique_s']
num_classes=graph_parameters['num_classes']

num_edges=num_nodes*(num_nodes-1)//2
input_dim=2


def loss_func(probs, cliques_r, cliques_s,num_classes=2):
    loss = 0
    # generate all possible pairs of nodes for a given number of nodes (i.e. all edges for a complete graph structure)
    # e.g. num_nodes=3, then output is tensor([[0,1],[0,2],[1,2]])
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    # create dictionary to map edges to their indices in the edge list, e.g. {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))} 
    edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()}) #add reverse edges
    
    # iterate over the batch of cliques of size r 
    # clique here is a tensor that contains the indices of the nodes 
    for clique in cliques_r:
        # extract all probabilities for an edge being blue
        probs_blue=probs[:,1]
        #generate all possible pairs of nodes within a clique and transpose, e.g. if a clique = torch.tensor([0, 1, 2, 3]),
        # then clique_edge will be tensor([[0, 0, 0, 1, 1, 2],[1, 2, 3, 2, 3, 3]])
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()] #get indices of edges in the clique
        edge_probs = probs_blue[edge_indices] #get probabilities of edges being blue in the clique
        blue_prod = edge_probs.prod() # product of these probs 
        loss += blue_prod
        
    # here everything is the same as above but for cliques of size s    
    for clique in cliques_s:
        probs_red=probs[:,0]
        clique_edge=torch.combinations(clique, r=2).t()
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        edge_probs = probs_red[edge_indices]
        red_prod = edge_probs.prod()
        loss += red_prod
    
    N = cliques_r.size(0) + cliques_s.size(0)
    
    return loss / N

# this is actually the loss function, but for a tensor probs[:,1], that I used for decoding solutions with conditional expectation
# since that would not work with the loss func as above because the tensor probs[:,1] contains only blue probabilities, I implemented additional loss
def cost_soft(probs_blue, cliques_r, cliques_s,num_classes=2):
    loss = 0
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    #dict to map edge to index
    edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))} 
    edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()}) #add reverse edges
    
    for clique in cliques_r:
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
    # I used this to track training time
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # sample a random batch of cliques of size r from the list of all cliques of size r
        cliques_r = random.sample(all_cliques_r.tolist(),batch_size)
        cliques_r=torch.tensor(cliques_r,dtype=torch.long)
        cliques_s = random.sample(all_cliques_s.tolist(),batch_size)
        cliques_s=torch.tensor(cliques_s,dtype=torch.long)

        # apply softmax to obtain a probability distribution over the colors, i.e. probs is a tensor where each row sums to 1 
        # (so that each row corresponds to a probability distribution over the two colors for an edge)
        probs=F.softmax(x, dim=1)
        loss = loss_func(probs, cliques_r, cliques_s,num_classes)
        loss.backward()
        optimizer.step()
        
        # log loss every 10 epochs and at the last epoch 
        # also probabilities from the last epoch to see if they are close to 1 or 0 
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            wandb.log({'epoch': epoch, 'loss': loss.item()})
        if epoch==num_epochs-1:
            prob_matrix=wandb.Table(data=probs.tolist(),columns=["red","blue"])
           
            wandb.log({"probs":prob_matrix})
    end_time = time.time()
    wandb.log({'runtime': end_time - start_time})
    return probs

# cost function, that counts how many blue cliques of size r and red cliques or size s are in the grapj
# input: edge_classes contains discretized probabilities
# edge_dict is a dictionary that contains all edges with their indices, e.g. {(0, 1): 0, (0, 2): 1, (1, 2): 2}
def cost_func(edge_classes, edge_dict, cliques_r, cliques_s):
    cost = 0
    
    # iterate over all cliques of size r 
    for clique in cliques_r:
        #generate all possible pairs of nodes within a clique and transpose, e.g. if a clique = torch.tensor([0, 1, 2, 3]),
        # then clique_edge will be tensor([[0, 0, 0, 1, 1, 2],[1, 2, 3, 2, 3, 3]])
        clique_edge = torch.combinations(clique, r=2).t()
        #get indices of edges in the clique
        edge_indices = [edge_dict[tuple(edge.tolist())] for edge in clique_edge.t()]
        # if edges are classified as blue (1), True, otherwise False
        blue_edges = edge_classes[edge_indices] == 1
        # if all edges are blue (i.e a clique is monochromatic), then add 1 to the cost
        if blue_edges.all():
            blue_prod = blue_edges.prod()
            cost += blue_prod
            
    # same as above but for red cliques   
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
        
        # generate all possible pairs of nodes for a given number of nodes (i.e. all edges for a complete graph structure)
        # e.g. num_nodes=3, then output is tensor([[0,1],[0,2],[1,2]])
        edge_list = torch.combinations(torch.arange(num_nodes), 2)
        # dict to map edge to index
        edge_dict = {tuple(edge_list[i].tolist()): i for i in range(edge_list.size(0))}
        edge_dict.update({(edge[1], edge[0]): i for edge, i in edge_dict.items()}) # add reverse edges

        # assign each edge to the class with the highest probability, hence index 1 for blue, 0 for red
        sets_thr = torch.argmax(probs, dim=1)
        # compute cost with sets_thr
        thresholded_cost = cost_func(sets_thr,edge_dict, cliques_r, cliques_s)
        # decode probabilities with the method of conditional expectation and then compute the cost 
        sets, cost= decode_graph(probs, edge_dict, cliques_r, cliques_s)
        # log both costs to wandb and then return the set that was decoded with conditional expectation
        wandb.log({'thresholded_cost':thresholded_cost, 'cost':cost})  
    return cost, sets

#retrieve solution with the method of conditional expectation
def decode_graph(probs, edge_dict, cliques_r, cliques_s):
    # extract only probabilities for an edge being blue (since then \P[edge is red]= 1 - \P[edge is blue])
    class_probs=probs[:,1]
    # sort the edges according to their probabilities in the decreasing order
    sorted_id = torch.argsort(class_probs, descending=True)
    sets = class_probs.detach().clone()
    
    # iterate over the sorted edge indices
    for idx in sorted_id:
        # create copies of the probabilties edge being blue to check both scenarios (p_{(i,j)}=1 and p_{(i,j)}=0)
        graph_probs_0 = sets.clone()
        graph_probs_1 = sets.clone()
        
        # set the current edge to red in one copy and blue in another
        graph_probs_0[idx] = 0  # Edge is red
        graph_probs_1[idx] = 1  # Edge is blue
        
        # compute the loss if the current edge is red
        expected_obj_0 = cost_soft(graph_probs_0, cliques_r, cliques_s)  
        # compute the loss if the current edge is blue
        expected_obj_1 = cost_soft(graph_probs_1, cliques_r, cliques_s)  
        
        # if the condition loss[edge blue]<loss[edge red] is met, i.e. the loss does not increase, set the edge to blue
        # otherwise red
        if expected_obj_0 > expected_obj_1:
            sets[idx] = 1  
            
        else:
            sets[idx] = 0 

    
    expected_obj_G = cost_func(sets, edge_dict, cliques_r, cliques_s)
    return sets, expected_obj_G  
        
    



def make_config(config):
    # given the number of nodes, generate all possible sets of nodes (cliques) of size clique_r
    cliques_r=torch.combinations(torch.arange(num_nodes),clique_r)
    # generate all possible sets of nodes (cliques) of size clique_s
    cliques_s=torch.combinations(torch.arange(num_nodes),clique_s)
    
    # generate input logits for the edges in the graph with values from either distribution, we will then turn these logits into probabilities 
    #x=torch.randn(num_edges,num_classes, requires_grad=True) #normal distribution
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


