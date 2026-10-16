"""
Train the Building-Diffusion model, conditioned on a trained (frozen) SSL program graph encoder.
    python scripts/train_diffusion.py [--config configs/diffusion.yaml] [section.key=value ...]
"""
import os
from datetime import datetime

import torch

from building_diffusion.config import (
    DiffusionConfig,
    config_parser,
    load_config,
    save_config,
)
from building_diffusion.constants import NUM_VOXEL_CLASSES
from building_diffusion.data.dataset import make_split_loaders
from building_diffusion.models.diffusion import (
    build_diffusion_model,
    default_ssl_checkpoint,
)
from building_diffusion.models.layers import position_encoder
from building_diffusion.models.noise_scheduler import NoiseScheduler
from building_diffusion.models.ssl import load_ssl_model_config
from building_diffusion.training.diffusion_loop import (
    complete_test,
    fit_diffusion,
    write_diffusion_log_header,
    write_test_results,
)
from building_diffusion.training.run import create_run
from building_diffusion.utils.plotting import (
    save_diffusion_plots,
    save_voxel_confusion_matrices,
)
from building_diffusion.utils.seed import set_seed


def main():
    print("="*100)
    print("Start Training New Model...")
    print("="*100)

    args = config_parser("Train the Building-Diffusion model.", "configs/diffusion.yaml").parse_args()
    cfg = load_config(DiffusionConfig, args.config, args.overrides)

    # Seed first: the data loaders and the model draw random numbers
    set_seed(cfg.train.seed, cfg.train.deterministic)

    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print([torch.cuda.get_device_name(0)] if torch.cuda.is_available() else "Using CPU")

    model_cfg = cfg.model
    hidden_channels = model_cfg.hidden_channels
    epochs = cfg.train.epochs
    # Resolve the default SSL checkpoint, so that config.yaml records the file actually used
    model_cfg.ssl_checkpoint = model_cfg.ssl_checkpoint or default_ssl_checkpoint(hidden_channels[0])
    # Architecture of the frozen SSL encoder, from the config.yaml of the SSL run
    ssl_model_cfg = load_ssl_model_config(model_cfg.ssl_checkpoint, hidden_channels[0])

    # Run folder: runs/dm_<date>_<layer>_<hidden size>_<number of layers>_<diffusion steps>_<epochs>
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_id = "_".join(["dm", current_time, model_cfg.layer_name, str(hidden_channels[0]), str(len(hidden_channels)), str(model_cfg.diffusion_steps), str(epochs)])
    model_folder, model_dir, writer_train, output_file = create_run(run_id, cfg)
    save_config(ssl_model_cfg, os.path.join(model_folder, "ssl_config.yaml"))
    write_diffusion_log_header(output_file, cfg)
    training_start_time = datetime.now()

    # Data: the complete test runs on batches of 32 buildings
    train_data_loader, val_data_loader, test_data_loader = make_split_loaders(cfg.data, cfg.train.batch_size, test_batch_size=32)

    # Model, floor positional encoding and noise scheduler
    diffusion_model = build_diffusion_model(model_cfg, device, model_cfg.ssl_checkpoint, ssl_model_cfg)
    print(diffusion_model)
    positional_encoding = position_encoder(device, hidden_dim=hidden_channels[0])
    noise_scheduler = NoiseScheduler(timesteps=model_cfg.diffusion_steps, x_classes=NUM_VOXEL_CLASSES, device=device)

    def run_complete_test(model, model_version):
        return complete_test(test_data_loader, model, device, torch.nn.CrossEntropyLoss(), noise_scheduler=noise_scheduler,
                             timesteps=model_cfg.diffusion_steps, model_folder=model_folder, model_version=model_version,
                             positional_encoding=positional_encoding, raw_dir=cfg.data.raw_dir)

    # Training
    history = fit_diffusion(diffusion_model, train_data_loader, val_data_loader, device, cfg.train.lr, epochs, noise_scheduler,
                            positional_encoding, model_dir, output_file, writer_train, run_complete_test)

    # Load the best model (the checkpoint already contains the SSL weights)
    output_file.write("Loading the best model!\n")
    diffusion_model = build_diffusion_model(model_cfg, device, ssl_model_cfg=ssl_model_cfg)
    diffusion_model.load_state_dict(torch.load(os.path.join(model_dir, "best_diffusion_model.pt"), map_location=device))
    diffusion_model.eval()
    diffusion_model.to(device)

    # Complete test of the best model: full reverse diffusion on the first test batch
    results = run_complete_test(diffusion_model, "best")
    write_test_results(output_file, results)
    _, _, _, test_true, test_pred, _, _, _, _ = results

    training_time = datetime.now() - training_start_time
    output_file.write("="*100)
    output_file.write('\nTraining Complete Duration: {}\n'.format(training_time))
    output_file.write('Best Epoch: {}\n'.format(history["best_epoch"]))
    output_file.write('Best Loss: {}\n'.format(history["best_loss"]))
    output_file.write('Best Accuracy: {:.2f}%\n'.format(history["best_accuracy"]))
    output_file.write("="*100)

    output_file.write("\nEnd date and time: {}\n".format(datetime.now()))
    output_file.close()
    writer_train.close()

    # Loss / accuracy plots and confusion matrices (best epoch train / validation, and test)
    save_diffusion_plots(model_folder, history, epochs)
    save_voxel_confusion_matrices(model_folder, history, test_true, test_pred)


if __name__ == "__main__":
    main()
