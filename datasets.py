# datasets.py
import torch
import random
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch

def inject_virtual_node(x, edge_index, edge_attr, y):
    """Injects a global ghost node and pads the target vector."""
    num_nodes = x.size(0)
    
    # 1. Add Virtual Node feature (zeros)
    v_node_feature = torch.zeros((1, x.size(1)), dtype=x.dtype)
    new_x = torch.cat([x, v_node_feature], dim=0)
    
    # 2. Bidirectional edges
    v_src = torch.arange(num_nodes)
    v_dst = torch.full((num_nodes,), num_nodes)
    v_edges = torch.cat([
        torch.stack([v_src, v_dst], dim=0),
        torch.stack([v_dst, v_src], dim=0)
    ], dim=1)
    new_edge_index = torch.cat([edge_index, v_edges], dim=1)
    
    # 3. Edge attributes (zeros)
    if edge_attr is not None:
        if edge_attr.dim() == 1:
            v_attr = torch.zeros((v_edges.size(1),), dtype=edge_attr.dtype)
        else:
            v_attr = torch.zeros((v_edges.size(1), edge_attr.size(1)), dtype=edge_attr.dtype)
        new_edge_attr = torch.cat([edge_attr, v_attr], dim=0)
    else:
        new_edge_attr = None
        
    # 4. Dummy target for the Virtual Node so loss calculations don't crash
    dummy_y = torch.zeros((1,), dtype=y.dtype)
    new_y = torch.cat([y, dummy_y], dim=0)
    
    return new_x, new_edge_index, new_edge_attr, new_y

class CumulativeXORDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.data = torch.randint(0, 2, (num_samples, seq_length), dtype=torch.long)
        self.labels = torch.zeros_like(self.data)
        self.labels[:, 0] = self.data[:, 0]
        for i in range(1, seq_length):
            self.labels[:, i] = self.labels[:, i-1] ^ self.data[:, i]

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class BFSDataset(Dataset):
    def __init__(self, num_samples, num_nodes):
        self.num_samples = num_samples
        self.graphs = []
        for _ in range(num_samples):
            G = nx.erdos_renyi_graph(n=num_nodes, p=0.3)
            while not nx.is_connected(G):
                G = nx.erdos_renyi_graph(n=num_nodes, p=0.3)
            source_node = torch.randint(0, num_nodes, (1,)).item()
            lengths = nx.single_source_shortest_path_length(G, source_node)
            
            x = torch.zeros((num_nodes, 1), dtype=torch.float)
            x[source_node] = 1.0
            y = torch.zeros((num_nodes,), dtype=torch.float)
            for node, dist in lengths.items(): y[node] = float(dist)
                
            edges = list(G.edges())
            edges += [(v, u) for u, v in edges] 
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.graphs.append(Data(x=x, edge_index=edge_index, y=y))
            
    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.graphs[idx]

class DijkstraDataset(Dataset):
    def __init__(self, num_samples, num_nodes, use_vn=False): # <-- Added use_vn
        self.num_samples = num_samples
        self.use_vn = use_vn # <-- Save the toggle state
        self.graphs = []
        
        # You pre-generate the graphs here (which is great for speed!)
        for _ in range(num_samples):
            G = nx.random_labeled_tree(num_nodes)
            for (u, v) in G.edges(): G.edges[u, v]['weight'] = random.randint(1, 10)
            source_node = random.randint(0, num_nodes - 1)
            lengths = nx.single_source_dijkstra_path_length(G, source_node, weight='weight')
            
            x = torch.zeros((num_nodes, 1), dtype=torch.float)
            x[source_node] = 1.0
            y = torch.zeros((num_nodes,), dtype=torch.float)
            for node, dist in lengths.items(): y[node] = dist
                
            edges, edge_weights = [], []
            for (u, v, data) in G.edges(data=True):
                w = data['weight']
                edges.extend([(u, v), (v, u)])
                edge_weights.extend([w, w])
                
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
            self.graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            
    def __len__(self): 
        return self.num_samples
        
    def __getitem__(self, idx):
        # 1. Unpack the pre-generated graph
        data = self.graphs[idx]
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        
        # 2. THE TOGGLE: Inject the Virtual Node dynamically if requested
        if self.use_vn:
            x, edge_index, edge_attr, y = inject_virtual_node(x, edge_index, edge_attr, y)
            
        # 3. Return the final graph
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class MaxSubarrayDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.data = []
        for _ in range(num_samples):
            sequence = torch.randint(-10, 11, (seq_length,), dtype=torch.float)
            y_seq = torch.zeros((seq_length,), dtype=torch.float)
            current_max = 0.0
            global_max = float('-inf')
            for i in range(seq_length):
                val = sequence[i].item()
                current_max = max(val, current_max + val)
                global_max = max(global_max, current_max)
                y_seq[i] = global_max
            self.data.append((sequence.view(-1, 1), y_seq.view(-1, 1)))
            
    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx]