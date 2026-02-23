import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import to_networkx
import networkx as nx

def forman_curvature(G):
    fc = {}
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        triangles = len(list(nx.common_neighbors(G, u, v)))
        degree_sum = G.degree[u] + G.degree[v]
        curvature = max(0.1, 4 - degree_sum + 3 * triangles)
        fc[key] = curvature
    return fc

def get_edge_curvature_tensor(data, device):
    G = to_networkx(data, to_undirected=True)
    forman = forman_curvature(G)

    curv_vals = []
    edge_index = data.edge_index.cpu()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        key = tuple(sorted((u, v)))
        curv_vals.append(forman.get(key, 0.0))

    return torch.tensor(curv_vals, dtype=torch.float, device=device)

class HeteroConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        diff = torch.abs(x_i - x_j)
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * self.lin(diff)
        return self.lin(diff)

class CurvatureGatedGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.homo_convs = torch.nn.ModuleList([
            GCNConv(in_dim, hidden_dim),
            GCNConv(hidden_dim, out_dim)
        ])
        self.hetero_convs = torch.nn.ModuleList([
            HeteroConv(in_dim, hidden_dim),
            HeteroConv(hidden_dim, out_dim)
        ])

    def forward(self, x, edge_index, edge_curvature):
        gate = torch.sigmoid(edge_curvature / 5.0).view(-1, 1)
        gate_homo = gate.squeeze()
        gate_hetero = (1 - gate).squeeze()

        for i in range(len(self.homo_convs)):
            is_last_layer = (i == len(self.homo_convs) - 1)

            x_homo = self.homo_convs[i](x, edge_index, edge_weight=gate_homo)
            
            x_hetero = self.hetero_convs[i](x, edge_index, edge_weight=gate_hetero)
            
            x = x_homo + x_hetero
            
            if not is_last_layer:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                
        return x
    

def train_model(model, data, edge_curv, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, edge_curv)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, edge_curv)
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

