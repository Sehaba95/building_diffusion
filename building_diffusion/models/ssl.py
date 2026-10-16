import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from building_diffusion.config import SSLConfig, SSLModelConfig, load_config
from building_diffusion.constants import (
    NUM_FLOORS,
    NUM_PROGRAM_CLASSES,
    PROGRAM_FEATURES,
    SSL_CONCAT,
    SSL_DROPOUT,
    SSL_ENCODING,
    SSL_HEADS,
    SSL_LAYER_NAME,
    SSL_NUM_LAYERS,
    SSL_OUT_CHANNELS,
)

from .layers import ConvBlock


class Encoder(nn.Module):
    """Program node features (N x in_channels) -> node embeddings (N x embedding_size), with graph convolutions."""
    def __init__(self, in_channels=PROGRAM_FEATURES, encoding = 32, 
                 hidden_channels=[32, 64, 128], embedding_size = 32, heads=8, 
                 dropout=0.5, layer_name="GATv2", concat=True):
        super(Encoder, self).__init__()

        # Preprocessing layers
        self.preprocessing_1 = nn.Linear(in_channels, encoding)
        self.preprocessing_2 = nn.Linear(encoding, encoding)

        # Convolution blocks
        self.encoder_convs = torch.nn.ModuleList()
        self.previous_hidden_channel = encoding

        for hidden_channel in hidden_channels:
            self.encoder_convs.append(ConvBlock(self.previous_hidden_channel, hidden_channel, layer_name=layer_name, heads=heads, dropout=dropout, concat=concat))
            if concat and layer_name in ["GATv2", "GAT", "TransformerConv"]:
                self.previous_hidden_channel = hidden_channel * heads
            else:
                self.previous_hidden_channel = hidden_channel

        # Last message-passing layer: always concat=False (heads averaged), so the node embeddings have embedding_size features
        self.encoder_convs.append(ConvBlock(self.previous_hidden_channel, embedding_size, layer_name=layer_name, heads=heads, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        x = F.relu(self.preprocessing_1(x))
        x = F.relu(self.preprocessing_2(x))

        for conv in self.encoder_convs:
            x = conv(x, edge_index)

        return x

class Decoder(nn.Module):
    """Node embeddings -> program class logits (N x 6) and floor logits (N x 11), with linear layers only."""
    def __init__(self, hidden_channels=[128, 64, 32], dropout=0.5, out_channels=32):
        super(Decoder, self).__init__()
        # Input size: the last encoder hidden channel (equal to embedding_size in the SSL configs)
        self.post_processing = nn.Linear(list(hidden_channels)[0], out_channels)

        # 6x node class, 11x node floor
        # Node Level predictions from the output of the decoder 
        self.class_classification = nn.Linear(out_channels, NUM_PROGRAM_CLASSES)  # Node classification
        self.floor_classification = nn.Linear(out_channels, NUM_FLOORS)  # Floor classification

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Decoder only have Linear layers (no message-passing is performed)
        x = F.relu(self.post_processing(x))
        
        x = self.dropout(x)

        output_class_classification = F.relu(self.class_classification(x))
        output_floor_classification = F.relu(self.floor_classification(x))

        return output_class_classification, output_floor_classification


class SSL(nn.Module):
    """
    Self-supervised program graph encoder: predicts the node classes and floors, and the building FAR and TPR
    from the mean-pooled graph embedding, which conditions the diffusion model.
    """
    def __init__(self, in_channels=PROGRAM_FEATURES, encoding = 32, hidden_channels=[32, 64, 128], embedding_size = 32, out_channels=32, heads=8, dropout=0.5, layer_name="GATv2", concat=True):
        super(SSL, self).__init__()
        self.num_graph_far = 1
        self.num_graph_tpr = NUM_PROGRAM_CLASSES
            
        # Encoder 
        self.encoder = Encoder(in_channels, encoding = encoding, hidden_channels=hidden_channels, embedding_size = embedding_size, heads=heads, dropout=dropout, layer_name=layer_name, concat=concat)
        
        # Embeddings
        # Graph Level Predictions from the output of the Encoder
        self.post_processing = nn.Linear(embedding_size, embedding_size)
        self.graph_far_preds = nn.Linear(embedding_size, self.num_graph_far) # Predict global_FAR 
        self.graph_tpr_preds = nn.Linear(embedding_size, self.num_graph_tpr) # Predict the target program ratios

        # Decoder 
        self.decoder = Decoder(reversed(hidden_channels), dropout=dropout, out_channels=out_channels)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x0, edge_index, batch=None):
        """
        x0: program node features (N x 19), edge_index: 2 x E, batch: graph index of every node (N).
        Returns None, the graph embeddings (B x embedding_size), FAR (B x 1) and TPR (B x 6) predictions,
        and the node class (N x 6) and floor (N x 11) logits.
        """
        # ENCODER 
        x = self.encoder(x0, edge_index)

        # Graph-level pooling : Graph Embedding
        graph_embeddings = global_mean_pool(x, batch)          
        # Transform to desired graph feature size        
        graph_features = self.dropout(F.relu(self.post_processing(graph_embeddings)))  
        # Predict the building FAR and target program ratios
        graph_far_predictions = F.relu(self.graph_far_preds(graph_features))  
        graph_tpr_predictions = F.relu(self.graph_tpr_preds(graph_features))  

        # DECODER
        output_class_classification, output_floor_classification = self.decoder(x)

        # The first output (floor embeddings) is not computed, but kept so SSL.forward still returns 6 values
        return None, graph_embeddings, graph_far_predictions, graph_tpr_predictions, output_class_classification, output_floor_classification


def build_ssl_model(model_cfg):
    """SSL from the model section of a config (SSLModelConfig); the embedding size is the last hidden size."""
    if model_cfg.in_channels != PROGRAM_FEATURES:
        raise ValueError(f"model.in_channels must be {PROGRAM_FEATURES} (program node features), got {model_cfg.in_channels}")
    return SSL(in_channels=model_cfg.in_channels,
               encoding=model_cfg.encoding,
               hidden_channels=model_cfg.hidden_channels,
               embedding_size=model_cfg.hidden_channels[-1],
               out_channels=model_cfg.out_channels,
               heads=model_cfg.heads,
               dropout=model_cfg.dropout,
               layer_name=model_cfg.layer_name,
               concat=model_cfg.concat)


def default_ssl_model_config(embedding_size):
    """Architecture of the SSL runs saved before config.yaml existed (constants.py)."""
    return SSLModelConfig(layer_name=SSL_LAYER_NAME, in_channels=PROGRAM_FEATURES, encoding=SSL_ENCODING,
                          hidden_channels=[embedding_size] * SSL_NUM_LAYERS, out_channels=SSL_OUT_CHANNELS,
                          heads=SSL_HEADS, concat=SSL_CONCAT, dropout=SSL_DROPOUT)


def load_ssl_model_config(ssl_checkpoint, embedding_size):
    """
    Architecture of a trained SSL model, from the config.yaml of its run (runs/<id>/checkpoints/<file>.pt -> runs/<id>/config.yaml).
    Runs without config.yaml fall back to default_ssl_model_config(embedding_size).
    """
    run_config_path = os.path.join(os.path.dirname(os.path.dirname(ssl_checkpoint)), "config.yaml")
    if not os.path.exists(run_config_path):
        print(f"No config.yaml found for {ssl_checkpoint}: using the default SSL architecture")
        return default_ssl_model_config(embedding_size)
    return load_config(SSLConfig, run_config_path).model
