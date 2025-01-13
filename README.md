## Computing Ramsey Numbers using Graph Neural Networks


### About this repository
There are three models implemented for constructing Ramsey graphs:
- **Baseline model,**
-  **Edge Prediction Network** (incorporating a neural network),
-   **Ramsey GNN** (incorporating a graph neural network).

The Baseline model implementation can be found in [baseline.py](baseline.py)

The Edge Prediction Network has two files: [edgenet.py](edgenet.py) includes the NN, loss and cost function implementation; [main.py](main.py) includes NN training and evaluation (i.e. decoding with conditional expectation or decoding by applying a threshold)

The same structure for the Ramsey GNN: [ramsey_gnn.py](ramsey_gnn.py) includes the GNN, loss and cost function; [mpnn_run.py](mpnn_run.py) includes GNN training and evaluation.

All results mentioned in the thesis can be found in my wandb project [gnn_ramsey](https://wandb.zib.de/ais2t/gnn-ramsey/sweeps). For each experiment, there is a sweep, named accordingly. 
