import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx

def compute_forman_curvature(data):
    """Calculates Forman curvature once and returns a dict."""
    G = to_networkx(data, to_undirected=True)
    fc = {}
    for u, v in G.edges():
        
        key = tuple(sorted((u, v)))
        triangles = len(list(nx.common_neighbors(G, u, v)))
        degree_sum = G.degree[u] + G.degree[v]
        curvature = max(0.1, 4 - degree_sum + 3 * triangles)
        fc[key] = curvature
    return fc


def add_curvature_data(data, mode='both'):
    """
    Assigns curvature as weights, features, or both.
    mode: 'weights', 'features', or 'both'
    """
    forman = compute_forman_curvature(data)
    edges = data.edge_index.t().tolist()
    
    vals = []
    for e in edges:
        key = tuple(sorted(e))
        vals.append(forman.get(key, 0.0))
    
    tensor = torch.tensor(vals, dtype=torch.float)
    

    def normalize(x, scale_min=0.1, scale_max=1.0):
        if x.max() > x.min():
            x = (x - x.min()) / (x.max() - x.min())
            return scale_min + (scale_max - scale_min) * x
        return torch.full_like(x, (scale_min + scale_max) / 2)

    if mode in ['weights', 'both']:
        data.edge_weight = normalize(tensor)
    
    if mode in ['features', 'both']:
        data.edge_curvature = normalize(tensor, 0.0, 1.0).unsqueeze(1)
        
    return data

class BaseCurvGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def _forward_common(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class StaticCurvGCN(BaseCurvGCN):
    def forward(self, x, edge_index, edge_weight):
        return self._forward_common(x, edge_index, edge_weight)

class LearnableCurvGCN(BaseCurvGCN):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.curv_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_curvature):
        edge_weight = self.curv_mlp(edge_curvature).squeeze(-1)
        edge_weight = 0.1 + 0.9 * edge_weight
        return self._forward_common(x, edge_index, edge_weight)