import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from building_diffusion.constants import NUM_PROGRAM_CLASSES, UNASSIGNED_LABEL
from building_diffusion.data.features import (
    floor_positional_encoding,
    program_graph,
    voxel_targets,
)
from building_diffusion.diffusion.sampling import reverse_diffusion, reverse_timesteps
from building_diffusion.metrics import (
    compute_far_distance,
    compute_far_tpr,
    compute_tpr_similarity,
)


def run_diffusion_epoch(data_loader, model, device, ce_loss, noise_scheduler, positional_encoding, optimizer=None):
    """
    One pass of the diffusion model over data_loader: trains when an optimizer is given, otherwise evaluates.
    Each batch is noised at a random timestep, and the model predicts the clean voxel labels (cross-entropy loss).
    Returns a dict with the mean loss, voxel accuracy (%), and the true / predicted voxel labels.
    """
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad()  # Clear gradients

    losses = []
    y_pred = []
    y_true = []
    num_nodes = 0
    correct_voxel_classification = 0

    with torch.set_grad_enabled(training):
        for data in data_loader:
            data = data.to(device)

            program_feat = program_graph(data)
            floor_pe = floor_positional_encoding(data, positional_encoding, device)

            # Generate the noisy graph using the NoiseScheduler
            g_0 = voxel_targets(data)
            timestep, g_T = noise_scheduler.apply_noise(g_0)
            timestep = timestep.to(device)

            # 14 Features : Dimension, coordinates, weight, g_T for noisy labels (6 classes + assigned status)
            node_features = torch.cat((data.voxel_feature, g_T.squeeze(0)), dim=1).clone().to(device)

            voxel_predictions = model(node_features, data.voxel_edge, data.voxel_feature_batch,
                                      timestep, program_feat, floor_pe=floor_pe)

            num_nodes += g_0.size(0)

            # Cross-entropy loss for node classification
            loss = ce_loss(voxel_predictions, g_0.argmax(dim=1))
            losses.append(loss.item())

            # Node classification accuracy
            voxel_predictions = voxel_predictions.argmax(dim=1)
            correct_voxel_classification += int((voxel_predictions == g_0.argmax(dim=1)).sum())
            y_true.extend(g_0.squeeze(0).cpu().argmax(dim=1).tolist())
            y_pred.extend(voxel_predictions.squeeze(0).cpu().tolist())

            if training:
                loss.backward()
                optimizer.step()  # Update parameters based on gradients
                optimizer.zero_grad()  # Clear gradients

    # The loss is the cross-entropy alone, so "loss" and "ce_loss" are the same value
    mean_loss = np.mean(losses)
    return {
        "loss": mean_loss,
        "ce_loss": mean_loss,
        "accuracy": (correct_voxel_classification / num_nodes) * 100,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _save_step_outputs(data, labels, timestep, model_folder, model_version, raw_dir):
    """For every graph of the batch, write the raw voxel JSON with the labels predicted at this timestep."""
    file_index = data.data_id_str
    for i, idx in enumerate(file_index):
        # Create a folder with index of the file
        output_file_path = os.path.join(str(model_folder), f"outputs_{model_version}", f"voxel_{idx}")
        os.makedirs(output_file_path, exist_ok=True)

        # Filter the voxels that belong to the current graph (i)
        mask = data.voxel_feature_batch == i

        # Load the original input JSON file, and save a copy next to the predictions
        input_file_path = os.path.join(raw_dir, "voxel_data", f"voxel_{idx}.json")
        with open(input_file_path, 'r') as file:
            input_data = json.load(file)
        with open(output_file_path+f"/voxel_data_{idx}.json", 'w') as json_file:
            json.dump(input_data, json_file, separators=(',', ':'))

        # Replace the true labels with the predicted labels of the current graph
        curr_voxel_pred = labels[mask].argmax(dim=1)
        for j, node in enumerate(input_data["voxel_node"]):
            if int(curr_voxel_pred[j]) == UNASSIGNED_LABEL:
                node['type'] = -1  # Unassigned voxel in the raw JSON format
            else:
                node['type'] = int(curr_voxel_pred[j])

        with open(output_file_path+f"/voxel_data_{idx}_timestep_{timestep}.json", 'w') as json_file:
            json.dump(input_data, json_file, separators=(',', ':'))


def complete_test(test_data_loader, model, device, ce_loss, noise_scheduler, timesteps, model_folder, model_version, positional_encoding, raw_dir):
    """
    Test the model by generating voxel labels with the full reverse diffusion process, starting from
    random labels, and comparing them with the ground truth. The JSON outputs of every timestep are
    saved for the first batch.
    Output:
        + model, loss, accuracy, true/predicted labels, FID (always 0), CE loss, TPR similarity, FAR distance
    """
    losses = []
    y_pred = []
    y_true = []
    tpr_similarities = []
    far_distances = []

    save_outputs = True
    num_nodes_test = 0
    correct_voxel_classification = 0
    test_fid = 0

    model.eval()
    schedule = reverse_timesteps(timesteps, device)

    with torch.no_grad():
        for data in tqdm(test_data_loader):
            data = data.to(device)

            program_feat = program_graph(data)
            floor_pe = floor_positional_encoding(data, positional_encoding, device)
            file_index = data.data_id_str

            # Save the true target values
            g_0 = voxel_targets(data)

            # Get True TPR and True FAR
            true_tpr = data.program_target_ratio.clone().view(len(file_index), NUM_PROGRAM_CLASSES)
            true_far = data.FAR.clone()

            num_nodes_test += g_0.size(0)

            on_step = None
            if save_outputs:
                def on_step(t, labels):
                    _save_step_outputs(data, labels, t, model_folder, model_version, raw_dir)

            predicted_graphs, voxel_predictions = reverse_diffusion(model, data, program_feat, floor_pe, noise_scheduler,
                                                                    schedule, device, on_step=on_step)

            # Only save the outputs of the first batch
            save_outputs = False

            # Known issue, kept for reproducibility: nodes are counted twice, which halves the reported accuracy
            num_nodes_test += predicted_graphs.size(0)

            # Compute TPR and FAR for predicted graph
            predicted_labels = predicted_graphs.argmax(dim=1)
            pred_far, pred_tpr = compute_far_tpr(file_index, data.voxel_feature_batch.max().item() + 1, predicted_labels, batch=data.voxel_feature_batch, device=device, raw_dir=raw_dir)

            # Calculate the TPR similarity and FAR distance
            far_distance = compute_far_distance(pred_far, true_far)
            tpr_similarity = compute_tpr_similarity(pred_tpr, true_tpr)

            # Loss of the last prediction vs true labels
            classification_loss = ce_loss(voxel_predictions, g_0.argmax(dim=1))

            # Accuracy of the sampled voxel graph
            correct_voxel_classification += int((predicted_labels == g_0.argmax(dim=1)).sum())
            y_true.extend(g_0.squeeze(0).cpu().argmax(dim=1).tolist())
            y_pred.extend(predicted_graphs.cpu().argmax(dim=1).tolist())

            losses.append(classification_loss.item())
            tpr_similarities.append(tpr_similarity.cpu().item())
            far_distances.append(far_distance.cpu().item())

            # Only the first test batch is sampled (the full test set is sampled by scripts/evaluate.py)
            break

    test_loss = np.mean(losses)
    ce_losses = np.mean(losses)
    tpr_similarity = np.mean(tpr_similarities)
    far_distance = np.mean(far_distances)

    test_accuracy = (correct_voxel_classification / num_nodes_test) * 100

    return model, test_loss, test_accuracy, y_true, y_pred, test_fid, ce_losses, tpr_similarity, far_distance


def write_diffusion_log_header(output_file, cfg):
    """Write the training settings at the top of the run log."""
    model_cfg = cfg.model
    output_file.write("Start date and time: {}\n".format(datetime.now()))
    output_file.write("Configuration details...\n")

    output_file.write(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Using CPU")
    output_file.write("="*100)

    output_file.write("\nTraining args:")
    output_file.write(f"\nModel name: {model_cfg.layer_name}")
    output_file.write(f"\nNumber of attention heads: {model_cfg.heads}")
    output_file.write(f"\nHidden_channels: {model_cfg.hidden_channels}")
    output_file.write(f"\nLearning rate: {cfg.train.lr}")
    output_file.write(f"\nNumber of epochs: {cfg.train.epochs}")
    output_file.write(f"\nBatch size: {cfg.train.batch_size}")
    output_file.write(f"\nSeed: {cfg.train.seed} (deterministic: {cfg.train.deterministic})\n")
    output_file.write("="*25)

    output_file.write("\nData args:")
    output_file.write(f"\nData directory: {cfg.data.processed_dir}")
    output_file.write(f"\nTrain size: {cfg.data.train_size}")
    output_file.write(f"\nTest size: {cfg.data.test_size}\n")
    output_file.write("="*25)

    output_file.write("\nModel args:")
    output_file.write(f"\nInput channels: {model_cfg.in_channels}")
    output_file.write(f"\nPre-processing embedding size: {model_cfg.encoding}")
    output_file.write(f"\nPost-processing embedding size: {model_cfg.out_channels}")
    output_file.write(f"\nConcatenate multi-head attention outputs : {model_cfg.concat}")
    output_file.write(f"\nResidual embedding concatenation: {model_cfg.residual}")
    output_file.write(f"\nTimestep embedding size: {model_cfg.timestep_embed_size}")
    output_file.write(f"\nDiffusion steps: {model_cfg.diffusion_steps}")
    output_file.write(f"\nSSL embedding size: {model_cfg.hidden_channels[0]}")
    output_file.write(f"\nSSL checkpoint: {model_cfg.ssl_checkpoint}")
    output_file.write(f"\nDropout: {model_cfg.dropout}\n")
    output_file.write(f"\nFloor Embedding: {model_cfg.floor_pos_enc}\n")

    output_file.write("=" *100 + "\n")


def write_test_results(output_file, results, epoch=None):
    """Log and print the results of complete_test (at a given epoch, or the final test)."""
    _, test_loss, test_accuracy, _, _, test_fid, test_ce_loss, test_tpr_loss, test_far_loss = results
    line = "Test Global Loss : {:.4f} | Test CE Loss : {:.4f} | Test TPR Loss : {:.4f} | Test FAR Loss : {:.4f} | Test Acc. : {:.4f}\n".format(
        test_loss, test_ce_loss, test_tpr_loss, test_far_loss, test_accuracy)

    if epoch is not None:
        output_file.write(f"Complete TEST Timesteps Diffusion at Epoch {epoch}\n")
        output_file.write(line)
        print("="*100)
        print(f"Complete Test at Epoch {epoch}")
        print(line)
    else:
        output_file.write("Complete TEST Timesteps Diffusion\n")
        output_file.write(line)
        output_file.write('\nTest Frechet Distance: {}\n'.format(test_fid))
        print("="*100)
        print(line)


def fit_diffusion(model, train_loader, val_loader, device, lr, epochs, noise_scheduler, positional_encoding,
                  model_dir, output_file, writer, run_complete_test):
    """
    Train the diffusion model with AdamW and ReduceLROnPlateau on the validation loss, for all epochs (no
    early stopping). The model with the lowest validation loss is saved to model_dir/best_diffusion_model.pt;
    every 100 epochs a checkpoint is saved and run_complete_test(model_version) is run.
    Returns a dict with the per-epoch losses / accuracies, the learning rates and the best epoch values.
    """
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=1e-4,
                                  betas=(0.9, 0.999))  # Adding momentum with beta_1=0.9

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-8)
    ce_loss = torch.nn.CrossEntropyLoss()

    history = {
        "train_losses": [], "train_accuracies": [], "val_losses": [], "val_accuracies": [], "learning_rates": [],
        "best_loss": float('inf'), "best_accuracy": -1, "best_epoch": -1,
        # Values for confusion matrices
        "best_train_true": None, "best_train_pred": None, "best_val_true": None, "best_val_pred": None,
    }

    optimizer.zero_grad()

    # No early stopping: every epoch is trained, and the model with the best validation loss is kept
    for epoch in range(epochs):
        epoch_start_time = datetime.now()

        output_file.write("\n\nEpoch {}/{}:\n".format(epoch + 1, epochs))
        output_file.write('================\n')

        print(f"Epoch {epoch + 1}/{epochs}:")
        print('==========================================')

        model = model.to(device)

        train = run_diffusion_epoch(train_loader, model, device, ce_loss, noise_scheduler, positional_encoding, optimizer=optimizer)
        val = run_diffusion_epoch(val_loader, model, device, ce_loss, noise_scheduler, positional_encoding)

        output_file.write("Train Global Loss : {:.4f} | Train CE Loss : {:.4f} Train Acc. : {:.4f}\n".format(
                        train["loss"], train["ce_loss"], train["accuracy"]))

        output_file.write("Val. Global Loss : {:.4f} | Val. CE Loss : {:.4f} Val. Acc. : {:.4f}\n".format(
                        val["loss"], val["ce_loss"], val["accuracy"]))

        print("Train Global Loss : {:.4f} | Train CE Loss : {:.4f} | Train Acc. : {:.4f}".format(
                        train["loss"], train["ce_loss"], train["accuracy"]))

        print("Val. Global Loss : {:.4f} | Val. CE Loss : {:.4f} |Val. Acc. : {:.4f}\n".format(
                        val["loss"], val["ce_loss"], val["accuracy"]))

        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)

        scheduler.step(val["loss"])
        output_file.write("Current Learning Rate : {:.4f}\n".format(current_lr))
        print("Current Learning Rate : {:.4f}\n".format(current_lr))

        history["train_losses"].append(train["loss"])
        history["train_accuracies"].append(train["accuracy"])
        history["val_losses"].append(val["loss"])
        history["val_accuracies"].append(val["accuracy"])

        # Update SummaryWriter
        writer.add_scalars(main_tag="Global Loss",
                           tag_scalar_dict={"train_loss": train["loss"], "val_loss": val["loss"]},
                           global_step=epoch)
        writer.add_scalars(main_tag="Node Accuracy",
                           tag_scalar_dict={"train_acc": train["accuracy"], "val_acc": val["accuracy"]},
                           global_step=epoch)

        # Calculate epoch duration
        epoch_duration = datetime.now() - epoch_start_time
        output_file.write("Epoch Duration: {}\n".format(epoch_duration))

        # Save the best model
        if val["loss"] < history["best_loss"]:
            history.update(best_loss=val["loss"], best_accuracy=val["accuracy"], best_epoch=epoch,
                           best_val_true=val["y_true"], best_val_pred=val["y_pred"],
                           best_train_true=train["y_true"], best_train_pred=train["y_pred"])
            torch.save(model.state_dict(), os.path.join(model_dir, "best_diffusion_model.pt"))

        if epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"diffusion_model_epoch_{epoch}.pt"))
            write_test_results(output_file, run_complete_test(model, f"epoch_{epoch}"), epoch=epoch)

    return history
