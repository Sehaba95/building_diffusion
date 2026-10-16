import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from building_diffusion.constants import NUM_PROGRAM_CLASSES, PROGRAM_CLASS_NAMES
from building_diffusion.data.features import build_program_features
from building_diffusion.utils.plotting import save_confusion_matrix

# Learning Rate Warmup Scheduler
WARMUP_EPOCHS = 5


def _warmup_factor(epoch):
    if (epoch + 1) <= WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1


def run_ssl_epoch(data_loader, model, device, ce_loss, mse_loss, optimizer=None):
    """
    One pass of the SSL model over data_loader: trains when an optimizer is given, otherwise evaluates.
    Loss = node class CE + floor CE + building FAR MSE + per-graph TPR MSE.
    Returns a dict with the mean global loss, the mean of each loss term, and node/floor accuracies (%).
    """
    training = optimizer is not None
    model.train(training)

    losses = []
    correct_node_classification = 0
    correct_floor_classification = 0
    num_nodes = 0

    node_classification_loss = 0
    floor_classification_loss = 0
    far_loss = 0
    tpr_loss = 0

    with torch.set_grad_enabled(training):
        for data in tqdm(data_loader):
            data = data.to(device)
            node_feat = build_program_features(data)

            _, _, far_predictions, tpr_predictions, node_classification, floor_classification = model(node_feat, data.program_edge, data.program_class_feature_batch)

            num_nodes += data.program_class_feature_batch.size()[0]

            # Loss calculation
            node_class = data.program_class_feature.argmax(dim=1)
            loss_1 = ce_loss(node_classification, node_class)  # Cross-entropy loss for node classification
            loss_2 = ce_loss(floor_classification, data.story_level_feature.long())  # Cross-entropy loss for floor classification
            loss_3 = mse_loss(far_predictions.squeeze(1), data.FAR)  # Mean-Squared Error loss for graph FAR
            loss_4 = mse_loss(tpr_predictions, data.program_target_ratio.view(data.num_graphs, NUM_PROGRAM_CLASSES))  # Mean-Squared Error loss for TPR per graph

            loss = loss_1 + loss_2 + loss_3 + loss_4
            losses.append(loss.item())

            node_classification_loss += loss_1.item()
            floor_classification_loss += loss_2.item()
            far_loss += loss_3.item()
            tpr_loss += loss_4.item()

            if training:
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.

            # Node and floor classification accuracy
            correct_node_classification += int((node_classification.argmax(dim=1) == node_class).sum())
            correct_floor_classification += int((floor_classification.argmax(dim=1) == data.story_level_feature).sum())

    return {
        "loss": sum(losses) / len(losses),
        "node_accuracy": 100 * correct_node_classification / num_nodes,
        "floor_accuracy": 100 * correct_floor_classification / num_nodes,
        "node_loss": node_classification_loss / len(losses),
        "floor_loss": floor_classification_loss / len(losses),
        "far_loss": far_loss / len(losses),
        "tpr_loss": tpr_loss / len(losses),
    }


def collect_ssl_predictions(data_loader, model, device):
    """Predicted and true node classes and floors over data_loader, for confusion matrices."""
    y_node_true, y_node_pred = [], []
    y_floor_true, y_floor_pred = [], []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            node_feat = build_program_features(data)
            _, _, _, _, node_classification, floor_classification = model(node_feat, data.program_edge, data.program_class_feature_batch)

            y_node_true.extend(data.program_class_feature.argmax(dim=1).tolist())
            y_node_pred.extend(node_classification.argmax(dim=1).tolist())
            y_floor_true.extend(data.story_level_feature.tolist())
            y_floor_pred.extend(floor_classification.argmax(dim=1).tolist())

    return y_node_true, y_node_pred, y_floor_true, y_floor_pred


def write_ssl_log_header(output_file, cfg, device_name):
    """Write the training settings at the top of the run log."""
    model_cfg = cfg.model
    output_file.write("Start date and time: {}\n".format(datetime.now()))
    output_file.write("Configuration details...\n")

    output_file.write(f"Using {device_name}\n")
    output_file.write("="*100)
    output_file.write("\nArgs:")
    output_file.write(f"\nModel name: {model_cfg.layer_name}")
    output_file.write(f"\nHidden_channel {model_cfg.hidden_channels}")
    output_file.write(f"\nLearning rate: {cfg.train.lr}")
    output_file.write(f"\nInput channels: {model_cfg.in_channels}")
    output_file.write(f"\nPreprocessing channels: {model_cfg.encoding}")
    output_file.write(f"\nEmbedding size (floor and graph embedding size): {model_cfg.hidden_channels[-1]}")
    output_file.write(f"\nPost-processing channels: {model_cfg.out_channels}")
    output_file.write(f"\nHeads: {model_cfg.heads}")
    output_file.write(f"\nConcatenate multi-head attention outputs : {model_cfg.concat}")
    output_file.write(f"\nDropout: {model_cfg.dropout}")
    output_file.write(f"\nData directory: {cfg.data.processed_dir}")
    output_file.write(f"\nTrain size: {cfg.data.train_size}")
    output_file.write(f"\nTest size: {cfg.data.test_size}")
    output_file.write(f"\nEpochs: {cfg.train.epochs}")
    output_file.write(f"\nBatch size: {cfg.train.batch_size}")
    output_file.write(f"\nSeed: {cfg.train.seed} (deterministic: {cfg.train.deterministic})\n")

    output_file.write("=" *100 + "\n")


def fit_ssl(model, train_loader, val_loader, device, lr, epochs, model_dir, output_file, writer):
    """
    Train the SSL model: AdamW with a linear learning-rate warmup (first WARMUP_EPOCHS epochs), then
    ReduceLROnPlateau on the validation loss, and early stopping after int(0.2 * epochs) epochs without
    improvement. The model with the lowest validation loss is saved to model_dir/best_ssl_model.pt.
    Returns the per-epoch train / validation metrics, the learning rates and the best epoch metrics.
    """
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=1e-4,
                                  betas=(0.9, 0.999))  # Adding momentum with beta_1=0.9

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8, verbose=True)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_factor)

    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    training_start_time = datetime.now()

    best_node_accuracy = -1
    best_floor_accuracy = -1
    best_epoch = -1

    # Metrics of every epoch (dicts returned by run_ssl_epoch)
    train_history = []
    val_history = []
    learning_rates = []

    # Early Stop
    patience = int(epochs * 0.2)  # Number of epochs with no improvement after which training will be stopped
    best_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(epochs):
        epoch_start_time = datetime.now()
        optimizer.zero_grad()

        output_file.write("\n\nEpoch {}/{}:\n".format(epoch + 1, epochs))
        output_file.write('================\n')

        print(f'Epoch {epoch + 1}/{epochs}:')
        print('==========')
        model.to(device)

        train = run_ssl_epoch(train_loader, model, device, ce_loss, mse_loss, optimizer=optimizer)
        val = run_ssl_epoch(val_loader, model, device, ce_loss, mse_loss)

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Learning Rate Warmup
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            print(f"WARMUP - Epoch {epoch} - Learning Rate: {current_lr}")
            output_file.write("Current Learning Rate : {:.4f}\n".format(current_lr))
        else:
            scheduler.step(val["loss"])
            output_file.write("Current Learning Rate : {:.4f}\n".format(current_lr))
            print("Current Learning Rate : {:.4f}\n".format(current_lr))

        train_history.append(train)
        val_history.append(val)

        writer.add_scalars(main_tag="Global Loss",
                           tag_scalar_dict={"train_loss": train["loss"], "test_loss": val["loss"]},
                           global_step=epoch)
        writer.add_scalars(main_tag="Node Accuracy",
                           tag_scalar_dict={"train_acc": train["node_accuracy"], "val_acc": val["node_accuracy"]},
                           global_step=epoch)
        writer.add_scalars(main_tag="Floor Accuracy",
                           tag_scalar_dict={"train_acc": train["floor_accuracy"], "val_acc": val["floor_accuracy"]},
                           global_step=epoch)

        # Save the best model
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_node_accuracy = val["node_accuracy"]
            best_floor_accuracy = val["floor_accuracy"]
            best_epoch = epoch
            no_improvement_epochs = 0  # Reset counter
            torch.save(model.state_dict(), os.path.join(model_dir, "best_ssl_model.pt"))
        else:
            no_improvement_epochs += 1  # Increment the counter
            # If the model is not improving, then stop training
            if no_improvement_epochs >= patience:
                print(f"EARLY STOPPING AT EPOCH: {epoch + 1}. THE MODEL STOPPED IMPROVING..")
                output_file.write(f"EARLY STOPPING AT EPOCH: {epoch + 1}. THE MODEL STOPPED IMPROVING..\n")
                break

        output_file.write("""Train Node Class. Loss : {:.4f} | Train Floor Class. Loss : {:.4f} | Train FAR Pred. Loss : {:.4f} | Train TPR Pred. Loss : {:.4f} \nVal. Node Class. Loss : {:.4f} | Val. Floor Class. Loss : {:.4f} | Val. FAR Pred. Loss : {:.4f} | Val. TPR Pred. Loss : {:.4f} \n""".format(
                    train["node_loss"], train["floor_loss"], train["far_loss"], train["tpr_loss"],
                    val["node_loss"], val["floor_loss"], val["far_loss"], val["tpr_loss"]))

        output_file.write("\nGlobal Train Loss : {:.4f} | Node Train Acc. : {:.4f} | Floor Train Acc. : {:.4f} \nGlobal Val. Loss : {:.4f} | Node Val. Acc. : {:.4f} | Floor Val. Acc. : {:.4f} \n".format(
                    train["loss"], train["node_accuracy"], train["floor_accuracy"], val["loss"], val["node_accuracy"], val["floor_accuracy"]))

        print("\nGlobal Train Loss : {:.4f} | Node Train Acc. : {:.4f} | Floor Train Acc. : {:.4f} |  \nGlobal Val. Loss : {:.4f} | Node Val. Acc. : {:.4f} | Floor Val. Acc. : {:.4f}\n".format(
                    train["loss"], train["node_accuracy"], train["floor_accuracy"], val["loss"], val["node_accuracy"], val["floor_accuracy"]))

        # Calculate epoch duration
        epoch_duration = datetime.now() - epoch_start_time
        print('Epoch Duration: {}'.format(epoch_duration))
        output_file.write("Epoch Duration: {}\n".format(epoch_duration))

    # Calculate the complete training time
    training_time = datetime.now()  - training_start_time
    print('Training Complete Duration: {}'.format(training_time))
    print("="*100)

    output_file.write("="*100)
    output_file.write('\nTraining Complete Duration: {}\n'.format(training_time))
    output_file.write('Best Epoch: {}\n'.format(best_epoch))
    output_file.write('Best Loss: {}\n'.format(best_loss))
    output_file.write('Best Node Accuracy: {:.2f}%\n'.format(best_node_accuracy))
    output_file.write('Best Floor Accuracy: {:.2f}%\n'.format(best_floor_accuracy))
    output_file.write("="*100)

    return train_history, val_history, learning_rates


def write_ssl_test_report(output_file, test):
    """Write the losses and accuracies of the best model (metrics dict from run_ssl_epoch)."""
    output_file.write("\nRESULTS USING TEST SET: \n")
    output_file.write("="*100)
    output_file.write("\nTest Node Class. Loss : {:.4f} | Test Floor Class. Loss : {:.4f} | Test FAR Pred. Loss : {:.4f} | Test TPR Pred. Loss : {:.4f} \n".format(
                    test["node_loss"], test["floor_loss"], test["far_loss"], test["tpr_loss"]))

    output_file.write("\nGlobal Test Loss : {:.4f} | Node Test Acc. : {:.4f} | Floor Test Acc. : {:.4f}\n".format(
                    test["loss"], test["node_accuracy"], test["floor_accuracy"]))


def save_ssl_confusion_matrices(run_path, model, loaders, device):
    """Node and floor classification confusion matrices for the train, validation and test loaders."""
    for (title, prefix), data_loader in zip((("Train", "Train"), ("Validation", "Val"), ("Test", "Test")), loaders):
        y_node_true, y_node_pred, y_floor_true, y_floor_pred = collect_ssl_predictions(data_loader, model, device)

        save_confusion_matrix(y_node_true, y_node_pred, os.path.join(run_path, f'{prefix}_Node_Classification_Confusion_Matrix.png'),
                              f"{title} Node Classification Confusion Matrix", display_labels=PROGRAM_CLASS_NAMES)
        save_confusion_matrix(y_floor_true, y_floor_pred, os.path.join(run_path, f'{prefix}_Floor_Classification_Confusion_Matrix.png'),
                              f"{title} Floor Classification Confusion Matrix")
