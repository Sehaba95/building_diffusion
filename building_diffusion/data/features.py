import torch
import torch_geometric

from building_diffusion.constants import FAR_SCALE, NUM_FLOORS


def build_program_features(data):
    """Program node features (N_program x 19): 6x node class, 11x node floor, 1x node TPR, 1x building FAR."""
    story_level = torch.nn.functional.one_hot(data.story_level_feature.long(), num_classes=NUM_FLOORS)
    node_tpr = data.program_target_ratio[data.program_class_cluster].unsqueeze(1)
    node_far = data.FAR[data.program_class_feature_batch].unsqueeze(1) * FAR_SCALE  # Normalize the FAR
    return torch.cat([data.program_class_feature, story_level, node_tpr, node_far], dim=1)


def program_graph(data):
    """Program graph inputs of the SSL encoder that conditions the diffusion model."""
    return {'x': build_program_features(data), 'edge_index': data.program_edge, 'batch': data.program_class_feature_batch}


def voxel_targets(data):
    """Ground-truth voxel labels (N_voxel x 7): one-hot over the 6 program classes + unassigned."""
    voxel_assigned = 1 - data.voxel_bool  # Convert voxel bool to Assigned or Unassigned
    return torch.cat((data.voxel_label, voxel_assigned.unsqueeze(1)), dim=1).clone()


def get_voxel_floor_level(voxel_floor_cluster, voxel_feature_batch, device):
    """
    Floor index of every voxel within its own building (N_voxel), from the batched floor cluster ids (N_voxel)
    and the voxel batch vector (N_voxel).
    """
    # Floor ids are serialized in the batch: subtract the lowest floor id of each graph
    pool = -torch_geometric.nn.max_pool_x(cluster=voxel_feature_batch,
                                          x=-voxel_floor_cluster,
                                          batch=torch.zeros(voxel_floor_cluster.shape[0], dtype=torch.long, device=device))[0]
    return voxel_floor_cluster-pool.index_select(0, voxel_feature_batch)


def floor_positional_encoding(data, positional_encoding, device):
    """Floor positional encoding of every voxel (N_voxel x hidden)."""
    voxel_floor_indices = get_voxel_floor_level(data.voxel_floor_cluster, data.voxel_feature_batch, device)
    return positional_encoding[voxel_floor_indices]
