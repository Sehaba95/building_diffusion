import torch
import torch.nn as nn
import torch.nn.functional as F

from building_diffusion.constants import NUM_VOXEL_CLASSES, VOXEL_FEATURES

from .layers import ConvBlock, SinusoidalEmbeddings
from .ssl import build_ssl_model, default_ssl_model_config

# Trained SSL models used by the paper's diffusion runs, by SSL embedding size (= hidden_channels[0])
DEFAULT_SSL_CHECKPOINTS = {
    64: "runs/May31_22-34-14_GATv2_5_64/checkpoints/best_ssl_model.pt",
    128: "runs/Jun01_02-41-20_GATv2_5_128/checkpoints/best_ssl_model.pt",
    256: "runs/Jun01_07-02-01_GATv2_5_256/checkpoints/best_ssl_model.pt",
}


def default_ssl_checkpoint(ssl_embedding_size):
    """Path of the trained SSL model for an embedding size (64, 128 or 256)."""
    if ssl_embedding_size not in DEFAULT_SSL_CHECKPOINTS:
        raise ValueError(
            f"No default SSL checkpoint for embedding size {ssl_embedding_size}. "
            f"Supported values are {list(DEFAULT_SSL_CHECKPOINTS)}; otherwise set model.ssl_checkpoint."
        )
    return DEFAULT_SSL_CHECKPOINTS[ssl_embedding_size]


class BuildingDiffusion(nn.Module):
    """
    ssl_checkpoint: trained SSL model loaded into the frozen SSL encoder. None skips loading, which is
    the case when a full diffusion checkpoint is loaded afterwards (it already contains the SSL weights).
    ssl_model_cfg: architecture of the frozen SSL encoder (SSLModelConfig); None uses default_ssl_model_config.
    """
    def __init__(self, device, in_channels=VOXEL_FEATURES, encoding=64, hidden_channels=[64, 64, 64, 64, 64],
                 out_channels=64, heads=8, dropout=0.5, layer_name=None, concat=False, residual=True,
                 timestep_embed_size=64, diffusion_steps=500, floor_pos_enc=True, ssl_checkpoint=None,
                 ssl_model_cfg=None):

        super(BuildingDiffusion, self).__init__()

        self.node_classes = NUM_VOXEL_CLASSES
        self.residual = residual
        self.device = device
        self.concat = concat
        self.floor_pos_enc = floor_pos_enc

        # The SSL embedding size is the size of the first hidden layer
        ssl_embedding_size = hidden_channels[0]

        # Frozen SSL encoder: same architecture as the trained SSL checkpoint
        ssl_model_cfg = ssl_model_cfg or default_ssl_model_config(ssl_embedding_size)
        if ssl_model_cfg.hidden_channels[-1] != ssl_embedding_size:
            raise ValueError(f"The SSL embedding size ({ssl_model_cfg.hidden_channels[-1]}, last value of the SSL hidden_channels) "
                             f"must equal the first value of the diffusion model's hidden_channels ({ssl_embedding_size})")
        self.ssl_model = build_ssl_model(ssl_model_cfg)

        # Load SSL model, and freeze it 
        if ssl_checkpoint is not None:
            self.ssl_model.load_state_dict(torch.load(ssl_checkpoint, map_location=device))
        self.ssl_model.eval()
        self.ssl_model.to(device)
        for param in self.ssl_model.parameters():
            param.requires_grad = False
    
        # Timestep Embedding
        self.timestep_encoder = SinusoidalEmbeddings(diffusion_steps=diffusion_steps, embed_dim=timestep_embed_size, device=self.device)
        
        ##############################################################################################################################33
        # Building-Diffusion layers
        self.embedding = ssl_embedding_size + timestep_embed_size 

        # Preprocessing layers
        encoder_input = in_channels + self.embedding
        self.preprocessing_1 = nn.Linear(encoder_input, encoding)
        self.preprocessing_2 = nn.Linear(encoding, hidden_channels[0])

        # Convolution blocks
        self.convs = torch.nn.ModuleList()
        self.previous_hidden_channel = hidden_channels[0]

        for hidden_channel in hidden_channels:

            self.convs.append(ConvBlock(self.previous_hidden_channel, hidden_channel, layer_name=layer_name, heads=heads, dropout=dropout, concat=self.concat))
            
            if self.concat:
                """
                If concat is True, the GATv2 layers will concatenate the output from each head
                Else the head's outputs will be averaged. 
                """
                self.previous_hidden_channel = hidden_channel * heads
                
            else:
                self.previous_hidden_channel = hidden_channel
        
        self.post_processing = nn.Linear(self.previous_hidden_channel, out_channels)

        # Output : one-hot class x7
        self.voxel_node_classification = nn.Linear(out_channels, self.node_classes)  # Node classification
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        ##############################################################################################################################33

    def train(self, mode=True):
        """Set the training mode, except for the frozen SSL encoder, which stays in eval mode (no dropout, fixed BatchNorm statistics)."""
        super().train(mode)
        self.ssl_model.eval()
        return self

    def forward(self, x, edge_index, batch, timestep, program_graph, floor_pe=None):
        """
        x: voxel features + noisy labels (N_voxel x 14), edge_index: voxel edges (2 x E), batch: graph index of
        every voxel (N_voxel), timestep: 1 x 1, program_graph: SSL inputs (see data.features.program_graph),
        floor_pe: floor positional encoding (N_voxel x hidden_channels[0]).
        Returns the predicted clean label logits (N_voxel x 7).
        """

        # Generate the Graph Embeddings using the trained SSL Model
        _, graph_embedding, _, _, _, _ = self.ssl_model(program_graph['x'], program_graph['edge_index'], program_graph['batch'])

        x = x.squeeze(0).float().to(self.device)
        edge_index = edge_index.to(self.device)

        # Timestep
        timestep_embedding = self.timestep_encoder(timestep).squeeze(1).repeat(x.shape[0], 1)
        
        # Get the number of voxels in each graph         
        _, counts = torch.unique(batch, return_counts=True)
        
        # Repeat each row (graph embedding) according to the number of voxels in each graph 
        graph_embedding = torch.repeat_interleave(graph_embedding, counts, dim=0)
        
        # The new embeddings: Graph Embedding, and Timestep Embedding
        embeddings = torch.cat((graph_embedding, timestep_embedding), dim=1)

        # Concatenate the embeddings with node features
        x = torch.cat((x, embeddings), dim=1)

        x = F.relu(self.preprocessing_1(x))
        x = F.relu(self.preprocessing_2(x))

        if self.floor_pos_enc:
            # Sum of pre-processed latent vectors and floor positional encoding
            x = x + floor_pe

        for _, conv in enumerate(self.convs):
            if self.residual:
                # Residual connection
                x = conv(x, edge_index) + x  
            else:
                x = conv(x, edge_index) 

        x = F.relu(self.post_processing(x))
        
        x = self.dropout(x)

        voxel_preds = self.voxel_node_classification(x)
        
        return voxel_preds



def build_diffusion_model(model_cfg, device, ssl_checkpoint=None, ssl_model_cfg=None):
    """BuildingDiffusion from the model section of a config (DiffusionModelConfig)."""
    if model_cfg.in_channels != VOXEL_FEATURES:
        raise ValueError(f"model.in_channels must be {VOXEL_FEATURES} (voxel features + noisy labels), got {model_cfg.in_channels}")
    return BuildingDiffusion(in_channels=model_cfg.in_channels,
                             encoding=model_cfg.encoding,  # Pre-Processing
                             hidden_channels=model_cfg.hidden_channels,
                             out_channels=model_cfg.out_channels,  # Post-Processing
                             heads=model_cfg.heads,
                             dropout=model_cfg.dropout,
                             concat=model_cfg.concat,
                             residual=model_cfg.residual,
                             device=device,
                             timestep_embed_size=model_cfg.timestep_embed_size,
                             diffusion_steps=model_cfg.diffusion_steps,
                             floor_pos_enc=model_cfg.floor_pos_enc,
                             layer_name=model_cfg.layer_name,  # GAT, GATv2, GCN, GatedGraphConv, TransformerConv, GIN, SAGE
                             ssl_checkpoint=ssl_checkpoint,
                             ssl_model_cfg=ssl_model_cfg)
