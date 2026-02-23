import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx

class GCN(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, use_edge_weight=False):
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if self.use_edge_weight and edge_weight is not None:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        if self.use_edge_weight and edge_weight is not None:
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        return x
    
    
def train_model(model, data, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, getattr(data, 'edge_weight', None))
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()



def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, getattr(data, 'edge_weight', None))
        pred = out.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = (pred[mask] == data.y[mask]).sum().item()
            total = mask.sum().item()
            acc = correct / total if total > 0 else 0.0
            accs.append(acc)
    return accs


