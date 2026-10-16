import os

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from building_diffusion.constants import NUM_VOXEL_CLASSES, VOXEL_CLASS_NAMES


def save_curves(path, series, title, xticks, yticks=None):
    """Save one figure with a line per entry of series ({label: values})."""
    plt.figure(figsize=(10, 7))
    for label, values in series.items():
        plt.plot(values, label=label)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_train_val(ax, train_values, val_values, title, xticks, yticks=None, labels=("Train", "Validation")):
    """Draw a train and a validation curve on a subplot axis."""
    ax.plot(train_values, label=labels[0])
    ax.plot(val_values, label=labels[1])
    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    ax.set_title(title)


def save_confusion_matrix(y_true, y_pred, path, title, display_labels=None, labels=None, normalize=None, **plot_kwargs):
    """Save the confusion matrix of the true / predicted class lists as an image."""
    cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(**plot_kwargs)
    disp.ax_.set_title(title)
    plt.savefig(path)
    plt.close()


def save_ssl_plots(run_path, train_history, val_history, learning_rates, epochs):
    """Loss / accuracy curves, learning rates and the 4 SSL losses, from the per-epoch metric dicts."""
    def series(history, key):
        return [metrics[key] for metrics in history]

    tick_step = max(1, int(epochs/5))

    figure, axis = plt.subplots(1, 3, figsize=(15, 10))
    plot_train_val(axis[0], series(train_history, "loss"), series(val_history, "loss"), "Global Loss",
                   xticks=range(0, epochs, tick_step), labels=("Training Loss", "Validation Loss"))
    plot_train_val(axis[1], series(train_history, "floor_accuracy"), series(val_history, "floor_accuracy"), "Floor Classification Accuracy",
                   xticks=range(0, epochs+1, tick_step), yticks=range(0, 101, 20), labels=("Training Accuracy", "Validation Accuracy"))
    plot_train_val(axis[2], series(train_history, "node_accuracy"), series(val_history, "node_accuracy"), "Node Classification Accuracy",
                   xticks=range(0, epochs+1, tick_step), yticks=range(0, 101, 20), labels=("Training Accuracy", "Validation Accuracy"))
    plt.savefig(os.path.join(run_path, 'Loss_Accuracies.png'))
    plt.close()

    save_curves(os.path.join(run_path, 'Learning_Rates_Plot.png'), {"Learning Rate Values": learning_rates}, "Learning Rate",
                xticks=range(0, epochs, tick_step))

    # Plot the 4 losses separately
    figure, axis = plt.subplots(1, 4, figsize=(15, 10))
    for ax, key, title in zip(axis, ("node_loss", "floor_loss", "far_loss", "tpr_loss"), ("Node Classif.", "Floor Classif.", "Node Pred.", "Graph Pred.")):
        plot_train_val(ax, series(train_history, key), series(val_history, key), title, xticks=range(0, epochs+1, tick_step))
    plt.savefig(os.path.join(run_path, 'Losses.png'))
    plt.close()


def save_diffusion_plots(run_path, history, epochs):
    """Loss, accuracy and learning-rate curves of the diffusion training (history from fit_diffusion)."""
    tick_step = max(1, int(epochs/5))
    save_curves(os.path.join(run_path, 'Loss_Plot.png'), {"Training Loss": history["train_losses"], "Validation Loss": history["val_losses"]},
                "Loss", xticks=range(0, epochs, tick_step))
    save_curves(os.path.join(run_path, 'Accuracy_Plot.png'), {"Training Accuracy": history["train_accuracies"], "Validation Accuracy": history["val_accuracies"]},
                "Classification Accuracy", xticks=range(0, epochs+1, tick_step), yticks=range(0, 101, 20))
    save_curves(os.path.join(run_path, 'Learning_Rate_Plot.png'), {"Learning Rate Values": history["learning_rates"]},
                "Learning Rate", xticks=range(0, epochs, tick_step))


def save_voxel_confusion_matrices(run_path, history, test_true, test_pred):
    """Voxel classification confusion matrices: train / validation of the best epoch, and the final test."""
    for y_true, y_pred, title, filename in ((history["best_train_true"], history["best_train_pred"], "Train Voxel Classification Confusion Matrix Epoch", 'Train_Voxel_Classification_Confusion_Matrix.png'),
                                            (history["best_val_true"], history["best_val_pred"], "Validation Voxel Classification Confusion Matrix", 'Validation_Voxel_Classification_Confusion_Matrix.png'),
                                            (test_true, test_pred, "Test Voxel Classification Confusion Matrix", 'Test_Voxel_Classification_Confusion_Matrix.png')):
        save_confusion_matrix(y_true, y_pred, os.path.join(run_path, filename), title, display_labels=VOXEL_CLASS_NAMES,
                              labels=list(range(NUM_VOXEL_CLASSES)), normalize='true', cmap='viridis', xticks_rotation='vertical')
