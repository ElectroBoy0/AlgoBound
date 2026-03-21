import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, MessagePassing

# ==========================================
# --- SEQUENCE MODELS ---
# ==========================================

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=2, d_model=64, nhead=4, num_layers=2, max_seq_len=10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(64)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        out = out + self.pos_embedding(positions)
        return self.fc_out(self.transformer_encoder(out))

# ==========================================
# ADVANCED SEQUENCE MODEL: RoPE Transformer
# ==========================================

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RoPE(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
    def forward(self, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]

class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RoPE(self.head_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, C))

class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class RoPETransformer(nn.Module):
    def __init__(self, vocab_size=2, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([RoPETransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size) 
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks: x = block(x)
        return self.fc_out(self.ln_f(x))

class SmallLSTM(nn.Module):
    def __init__(self, vocab_size=2, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))
        return self.fc_out(lstm_out)

class KadaneLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)

# ==========================================
# --- GRAPH MODELS ---
# ==========================================

class SimpleMPNN(nn.Module):
    def __init__(self, node_feature_dim=1, hidden_dim=32, num_classes=1):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(node_feature_dim if i==0 else hidden_dim, hidden_dim) for i in range(4)])
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.fc(x).squeeze(-1)

class MinMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='min')
        self.message_mlp = nn.Sequential(nn.Linear(node_dim + edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(nn.Linear(node_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=1))
    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=1))

class DijkstraGNN(nn.Module):
    def __init__(self, node_feature_dim=1, edge_feature_dim=1, hidden_dim=64, num_layers=5):
        super().__init__()
        self.node_emb = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList([MinMPNNLayer(hidden_dim, edge_feature_dim, hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        for layer in self.layers: x = layer(x, edge_index, edge_attr)
        return self.fc(x).squeeze(-1)

# ==========================================
# --- STATE SPACE MODELS (MAMBA) ---
# ==========================================

class SimplifiedMambaBlock(nn.Module):
    """
    A pure-PyTorch implementation of a Selective State Space Model.
    Stabilized with negative state initializations and proper exponential discretization.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # FIX 1: Stable Initialization. 'A' must be strictly negative to prevent exploding states!
        self.A = nn.Parameter(-torch.rand(d_model, d_state) - 0.5) 
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1) 
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        
        self.delta = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :] 
            dt = F.softplus(self.delta(xt)) 
            dt_unsqueeze = dt.unsqueeze(-1) 
            
            # FIX 2: Proper Discretization using torch.exp() guarantees values stay bounded
            A_bar = torch.exp(dt_unsqueeze * self.A.unsqueeze(0)) 
            B_bar = dt_unsqueeze * self.B(xt).unsqueeze(1)      
            
            h = A_bar * h + B_bar
            yt = (h * self.C).sum(dim=-1) + self.D * xt
            outputs.append(yt.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)

class MambaModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_layers=2):
        super().__init__()
        # Using Embedding to align with how your Transformers handle the XOR vocab
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        
        # Stack the Mamba blocks
        self.layers = nn.ModuleList([SimplifiedMambaBlock(hidden_dim) for _ in range(num_layers)])
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is expected to be token indices (batch_size, seq_len)
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        # Extract the sequence states for classification
        return self.fc(x)