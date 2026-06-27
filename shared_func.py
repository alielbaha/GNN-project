import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_networkx, add_self_loops
import networkx as nx



device = torch.device('cusda' if torch.cuda.is_available() else 'cpu')

def forman_curvature(G): 
    fc = {}
    for u, v in G.edges():
        triangles = len(list(nx.common_neighbors(G, u, v)))
        fc[(u, v)] = 4 - (G.degree[u] + G.degree[v]) + 3 * triangles
    return fc

def add_forman_edge_weights(data, normalize=True): 

    G = to_networkx(data, to_undirected=True)
    fc = forman_curvature(G)
    
    edge_index = data.edge_index.cpu()
    num_edges = edge_index.shape[1]
    weights = []
    
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        key = tuple(sorted((src, dst)))
        weights.append(fc.get(key, 0.0))
        
    weights = torch.tensor(weights, dtype=torch.float)
    
    if normalize:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
            weights = 0.1 + 0.9 * weights  
        else:
            weights = torch.full_like(weights, 0.5)
            
    data.edge_weight = weights
    return data

def get_random_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    n = data.num_nodes
    idx = torch.randperm(n)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    
    data.train_mask[idx[:train_size]] = True
    data.val_mask[idx[train_size:train_size + val_size]] = True
    data.test_mask[idx[train_size + val_size:]] = True
    return data

class HeteroConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True, normalize=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)
        

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0)
            )
            
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(0), add_self_loops=False
            )
            
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j