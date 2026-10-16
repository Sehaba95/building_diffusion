import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from building_diffusion.constants import (
    FGW_MIN_ALPHA,
    NUM_PROGRAM_CLASSES,
    UNASSIGNED_LABEL,
)


def compute_far_tpr(data_id_str, batch_size, labels, batch, device, raw_dir):
    """
    FAR and target program ratios (TPR) of generated buildings, from the voxel weights of the raw voxel JSON files.
        data_id_str: building ids of the batch (B)
        labels: predicted voxel class indices (N_voxel), UNASSIGNED_LABEL for unassigned voxels
        batch: graph index of every voxel (N_voxel)
    Returns the FAR (B) and the TPR (B x 6), normalized to sum to 1 per building.
    """
    # Initialize the values of FAR and TPR 
    pred_far = torch.zeros(batch_size, device=device)
    pred_tpr = torch.zeros(batch_size, NUM_PROGRAM_CLASSES, device=device)
    
    # Load the voxel data file and calculate the FAR and TPR 
    for i, file_idx in enumerate(data_id_str):
        mask = (batch == i)
        
        file_path = os.path.join(raw_dir, "voxel_data", f"voxel_{file_idx}.json")

        # Load the original input JSON file
        with open(file_path, 'r') as file:
            json_file = json.load(file)
        
        for node, label in zip(json_file["voxel_node"], labels[mask]):
            # If the voxel is not unassigned, add its weight to the FAR and to the TPR of its program
            if label != UNASSIGNED_LABEL:
                pred_far[i] += node['weight']
                pred_tpr[i][label] += node['weight']

        # Normalize TPR to be summed to 1
        if pred_far[i] > 0:
            pred_tpr[i] = pred_tpr[i] / pred_far[i]
        else:
            pred_tpr[i] = torch.zeros_like(pred_tpr[i])
    
    return pred_far, pred_tpr 

def compute_far_distance(pred_far, target_far):
    """Sum over the batch of the relative FAR error |pred_far - target_far| / target_far (B)."""

    # Compute the absolute difference between actual and target FAR
    abs_difference = torch.abs(pred_far - target_far)
    
    # Normalize the difference by the target FAR
    far_distance = abs_difference / target_far
    
    return far_distance.sum()


def compute_tpr_similarity(actual_ratios, target_ratios):
    """
    Sum over the batch of the cosine similarities between the generated and target program ratios (TPR).
    
    Args:
        actual_ratios (torch.Tensor): Actual ratios of room types (shape: [batch_size, num_room_types]).
        target_ratios (torch.Tensor): Target ratios of room types (same shape as actual_ratios).
    
    Returns:
        torch.Tensor: Sum of the cosine similarities (scalar).
    """
    # Compute cosine similarity (requires input to be at least 2D)
    cosine_sim = F.cosine_similarity(actual_ratios, target_ratios, dim=1)
    
    return cosine_sim.sum()

def compute_fgw_distance(C1, C2, X1, X2, alpha=0.5):
    """
    Compute Fused Gromov-Wasserstein distance between two graphs
    Inputs:
      C1, C2: intra-graph cost matrices (n1 x n1, n2 x n2)
      X1, X2: node feature matrices (n1 x d, n2 x d)
      alpha: trade-off between structure and features (0: features only, replaced by FGW_MIN_ALPHA)
    Returns:
      fgw_dist: FGW distance (scalar)
    Reference: 
      https://arxiv.org/pdf/1805.09114
    """
    # Imported here so that only FGW evaluation requires POT, not training
    import ot

    # uniform node distributions
    a1 = np.ones((C1.shape[0],)) / C1.shape[0]
    a2 = np.ones((C2.shape[0],)) / C2.shape[0]
    
    # cross-feature cost (squared euclidean)
    M = ot.dist(X1, X2, metric="sqeuclidean")
    
    # POT divides by alpha to log the structure term: alpha = 0 gives a 0 / 0 warning
    if alpha == 0:
        alpha = FGW_MIN_ALPHA

    # FGW
    fgw_dist, _ = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, a1, a2, loss_fun="square_loss",
        alpha=alpha, armijo=False, log=True
    )
    return fgw_dist
