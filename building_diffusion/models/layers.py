import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GatedGraphConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
)

from building_diffusion.constants import MAX_FLOORS_POS_ENC


def _gin_conv(in_channels, out_channels):
    nn_gin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
    return GINConv(nn_gin)


# layer_name -> constructor(in_channels, out_channels, heads, concat)
GRAPH_CONV_LAYERS = {
    "GAT": lambda i, o, heads, concat: GATConv(i, o, heads=heads, concat=concat),
    "GATv2": lambda i, o, heads, concat: GATv2Conv(i, o, heads=heads, concat=concat),
    "GCN": lambda i, o, heads, concat: GCNConv(i, o),
    "GatedGraphConv": lambda i, o, heads, concat: GatedGraphConv(o, num_layers=1),  # num_layers can be adjusted
    "TransformerConv": lambda i, o, heads, concat: TransformerConv(i, o, heads=heads, concat=concat),
    "GIN": lambda i, o, heads, concat: _gin_conv(i, o),
    "SAGE": lambda i, o, heads, concat: SAGEConv(i, o),
}

# Layers whose output has out_channels * heads features when concat=True
MULTI_HEAD_LAYERS = ("GAT", "GATv2", "TransformerConv")


class ConvBlock(nn.Module):
    """Graph convolution -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, in_channels, out_channels, heads, dropout, concat, layer_name="GATv2"):
        super(ConvBlock, self).__init__()

        if layer_name not in GRAPH_CONV_LAYERS:
            raise ValueError(f"Unknown layer_name {layer_name!r}. Expected one of {list(GRAPH_CONV_LAYERS)}.")

        self.conv = GRAPH_CONV_LAYERS[layer_name](in_channels, out_channels, heads, concat)
        if concat and layer_name in MULTI_HEAD_LAYERS:
            self.bn = nn.BatchNorm1d(out_channels * heads)
        else:
            self.bn = nn.BatchNorm1d(out_channels)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn(self.conv(x, edge_index)))
        x = self.dropout(x)
        return x


def sinusoidal_table(length, dim, device=None):
    """Transformer-style sinusoidal encodings: row p holds sin/cos(p / 10000^(2i/dim)). Shape: length x dim."""
    table = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table


class SinusoidalEmbeddings(nn.Module):
    """Diffusion timestep embeddings (diffusion_steps x embed_dim)."""

    def __init__(self, diffusion_steps: int, embed_dim: int, device):
        super().__init__()
        self.device = device
        # Built on the CPU, then moved. Kept as a plain attribute (not a buffer) so that checkpoint keys are unchanged.
        self.embeddings = sinusoidal_table(diffusion_steps, embed_dim).to(self.device)

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)

        t = t.long().to(self.device)
        embeds = self.embeddings[t].to(self.device)
        return embeds


def position_encoder(device, max_len=MAX_FLOORS_POS_ENC, hidden_dim=128):
    """Floor positional encoding (max_len x hidden_dim), built directly on `device`."""
    return sinusoidal_table(max_len, hidden_dim, device=device)
