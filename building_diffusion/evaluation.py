import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from building_diffusion.config import DiffusionModelConfig, SSLModelConfig, build_config
from building_diffusion.constants import NUM_FLOORS, NUM_PROGRAM_CLASSES, VOXEL_FEATURES
from building_diffusion.data.features import floor_positional_encoding, program_graph
from building_diffusion.diffusion.sampling import reverse_diffusion, reverse_timesteps
from building_diffusion.metrics import (
    compute_far_distance,
    compute_far_tpr,
    compute_fgw_distance,
    compute_tpr_similarity,
)
from building_diffusion.models.ssl import (
    default_ssl_model_config,
    load_ssl_model_config,
)


def complete_test_evaluation(test_data_loader, model, device, noise_scheduler, timesteps, positional_encoding, raw_dir, num_inferences=1):
    """
    Evaluate the model on the test set (batch size 1). For every building, generate num_inferences
    samples with the full reverse diffusion process, keep the sample with the lowest FGW distance
    to the ground truth, and average its metrics over the test set.
    Output:
        + TPR similarity, FAR distance, Fused Gromov-Wasserstein (FGW) distance
    """
    fgw_distances = []
    tpr_similarities = []
    far_distances = []

    model.eval()
    schedule = reverse_timesteps(timesteps, device)

    with torch.no_grad():
        for data in tqdm(test_data_loader):
            data = data.to(device)

            program_feat = program_graph(data)
            floor_pe = floor_positional_encoding(data, positional_encoding, device)
            file_index = data.data_id_str

            # Get True TPR and True FAR
            true_tpr = data.program_target_ratio.clone().view(len(file_index), NUM_PROGRAM_CLASSES)
            true_far = data.FAR.clone()

            # Evaluation metrics of every sample of the current building
            curr_fgw_distances = []
            curr_tpr_similarities = []
            curr_far_distances = []

            for inference_idx in range(num_inferences):
                predicted_graphs, _ = reverse_diffusion(model, data, program_feat, floor_pe, noise_scheduler, schedule, device)

                # Compute TPR and FAR for predicted graph
                predicted_labels = predicted_graphs.argmax(dim=1)
                pred_far, pred_tpr = compute_far_tpr(file_index, data.voxel_feature_batch.max().item() + 1, predicted_labels, batch=data.voxel_feature_batch, device=device, raw_dir=raw_dir)

                # Calculate the TPR similarity and FAR distance
                far_distance = compute_far_distance(pred_far, true_far)
                tpr_similarity = compute_tpr_similarity(pred_tpr, true_tpr)

                # Fused Gromov-Wasserstein distance between the true and the predicted voxel graph
                num_nodes = int(data.voxel_edge.max()) + 1

                # Generation only changes the labels: both graphs share the voxel adjacency
                adj = to_dense_adj(data.voxel_edge, max_num_nodes=num_nodes).squeeze(0).cpu().numpy()

                voxel_floor = F.one_hot(data.voxel_floor_cluster, num_classes=NUM_FLOORS)
                X_gt = torch.cat([data.voxel_label, voxel_floor], dim=-1)
                X_pred = torch.cat([predicted_graphs[:, :NUM_PROGRAM_CLASSES], voxel_floor], dim=-1)  # The predicted program labels

                fgw_distance = compute_fgw_distance(adj,
                                                    adj,
                                                    X_gt.cpu().numpy(),
                                                    X_pred.cpu().numpy(),
                                                    alpha=0)

                curr_fgw_distances.append(fgw_distance)
                curr_tpr_similarities.append(tpr_similarity)
                curr_far_distances.append(far_distance)

            # Keep the sample with the minimum FGW distance
            fgw_argmin = np.argmin(curr_fgw_distances)
            fgw_distances.append(curr_fgw_distances[fgw_argmin])
            tpr_similarities.append(curr_tpr_similarities[fgw_argmin].cpu().item())
            far_distances.append(curr_far_distances[fgw_argmin].cpu().item())

    tpr_similarity = np.mean(tpr_similarities)
    far_distance = np.mean(far_distances)
    normalized_fgw_distance = np.mean(fgw_distances)

    return tpr_similarity, far_distance, normalized_fgw_distance


def extract_flags_from_log(model_dir):
    """Old runs (without config.yaml): read the layer name, residual and floor flags from the training log."""
    # Find the first log file matching model_dm*.log
    log_files = glob.glob(os.path.join(model_dir, "model_dm*.log"))
    if not log_files:
        raise FileNotFoundError("No log files matching 'model_dm*.log' found in " + model_dir)

    log_file = log_files[0]  # take the first match
    layer_name = None
    residual_flag = False
    floor_flag = False

    # Read the log file line by line
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip().lower().startswith("model name:"):
                layer_name = line.split(":", 1)[1].strip()
            line = line.strip().lower()  # lowercase for case-insensitive comparison
            if "residual embedding concatenation" in line:
                if "true" in line:
                    residual_flag = True
            if "floor embedding" in line:
                if "true" in line:
                    floor_flag = True

    return layer_name, residual_flag, floor_flag


def legacy_model_config(run_path, layer_name):
    """Settings of the runs trained before config.yaml existed (hard-coded in the original evaluation script)."""
    log_layer_name, residual, floor_pos_enc = extract_flags_from_log(run_path)
    layer_name = layer_name or log_layer_name
    if layer_name is None:
        raise ValueError(f"The layer name of {run_path} is not in its log: set it with eval.layer_name=<name>")
    return DiffusionModelConfig(layer_name=layer_name, in_channels=VOXEL_FEATURES, encoding=256, hidden_channels=[256, 256, 256],
                                out_channels=256, heads=8, concat=False, residual=residual, floor_pos_enc=floor_pos_enc,
                                dropout=0.5, timestep_embed_size=256, diffusion_steps=500, ssl_checkpoint=None)


def load_run_model_config(run_path, layer_name=None):
    """
    Model settings of a training run: from runs/<id>/config.yaml, or for older runs without it, from
    legacy_model_config (layer_name, if given, replaces the layer name read from the log).
    """
    run_config_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(run_config_path):
        print(f"No config.yaml in {run_path}: using the settings of the original evaluation script")
        return legacy_model_config(run_path, layer_name)

    if layer_name is not None:
        print(f"Warning: eval.layer_name is ignored, the layer name is read from {run_config_path}")
    with open(run_config_path, "r") as file:
        return build_config(DiffusionModelConfig, yaml.safe_load(file)["model"], "model.")


def load_run_ssl_model_config(run_path, model_cfg):
    """
    Architecture of the frozen SSL encoder of a training run: from runs/<id>/ssl_config.yaml, or for older runs
    without it, from the config.yaml of the SSL run (model_cfg.ssl_checkpoint), or else the default SSL architecture.
    """
    ssl_config_path = os.path.join(run_path, "ssl_config.yaml")
    if os.path.exists(ssl_config_path):
        with open(ssl_config_path, "r") as file:
            return build_config(SSLModelConfig, yaml.safe_load(file))
    if model_cfg.ssl_checkpoint is not None:
        return load_ssl_model_config(model_cfg.ssl_checkpoint, model_cfg.hidden_channels[0])
    return default_ssl_model_config(model_cfg.hidden_channels[0])


def write_model_details(output_file, model_cfg, num_classes):
    """Write the model settings to the evaluation log."""
    output_file.write("\nDiffusion Model Details\n")
    output_file.write(f"Concatenate Attention Heads: {model_cfg.concat}\n")
    output_file.write(f"Residual Connections: {model_cfg.residual}\n")
    output_file.write(f"Floor Positional Encoding: {model_cfg.floor_pos_enc}\n")
    output_file.write(f"Input Features: {model_cfg.in_channels}\n")
    output_file.write(f"Output Classes: {num_classes}\n")
    output_file.write(f"Timesteps: {model_cfg.diffusion_steps}\n")
    output_file.write(f"Encoding Size: {model_cfg.encoding}\n")
    output_file.write(f"Hidden Space: {model_cfg.hidden_channels}\n")
    output_file.write(f"Heads: {model_cfg.heads}\n")
    output_file.write(f"Dropout: {model_cfg.dropout}\n")
    output_file.write(f"Timestep Embedding Size: {model_cfg.timestep_embed_size}\n")
    output_file.write(f"SSL Embedding Size: {model_cfg.hidden_channels[0]}\n")
    output_file.write("="*25)
