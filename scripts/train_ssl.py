"""
Train the self-supervised program graph encoder (SSL). It is frozen afterwards and conditions the diffusion model.
    python scripts/train_ssl.py [--config configs/ssl.yaml] [section.key=value ...]
"""
import os
from datetime import datetime

import torch

from building_diffusion.config import SSLConfig, config_parser, load_config
from building_diffusion.data.dataset import make_split_loaders
from building_diffusion.models.ssl import build_ssl_model
from building_diffusion.training.run import create_run
from building_diffusion.training.ssl_loop import (
    fit_ssl,
    run_ssl_epoch,
    save_ssl_confusion_matrices,
    write_ssl_log_header,
    write_ssl_test_report,
)
from building_diffusion.utils.plotting import save_ssl_plots
from building_diffusion.utils.seed import set_seed


def main():
    args = config_parser("Train the self-supervised program graph encoder.", "configs/ssl.yaml").parse_args()
    cfg = load_config(SSLConfig, args.config, args.overrides)

    # Seed first: the data loaders and the model draw random numbers
    set_seed(cfg.train.seed, cfg.train.deterministic)

    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    print(f"Using {device_name}")

    model_cfg = cfg.model
    hidden_channels = model_cfg.hidden_channels
    epochs = cfg.train.epochs

    # Run folder: runs/<date>_<layer>_<number of layers>_<embedding size>
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_id = "_".join([current_time, model_cfg.layer_name, str(len(hidden_channels)), str(hidden_channels[-1])])
    run_path, model_dir, writer_train, output_file = create_run(run_id, cfg)
    write_ssl_log_header(output_file, cfg, device_name)

    # Data and model
    loaders = make_split_loaders(cfg.data, cfg.train.batch_size, test_batch_size=cfg.train.batch_size)
    train_data_loader, val_data_loader, test_data_loader = loaders
    model = build_ssl_model(model_cfg)

    # Training
    train_history, val_history, learning_rates = fit_ssl(model, train_data_loader, val_data_loader, device, cfg.train.lr,
                                                         epochs, model_dir, output_file, writer_train)

    # Load the best model
    output_file.write("Loading the best model for test!\n")
    model = build_ssl_model(model_cfg)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_ssl_model.pt"), map_location=device))
    model.eval()
    model.to(device)

    test = run_ssl_epoch(test_data_loader, model, device, torch.nn.CrossEntropyLoss(), torch.nn.MSELoss())
    write_ssl_test_report(output_file, test)

    output_file.write("\nEnd date and time: {}\n".format(datetime.now()))
    output_file.close()
    writer_train.close()

    # Loss / accuracy plots and confusion matrices (train, validation, test)
    save_ssl_plots(run_path, train_history, val_history, learning_rates, epochs)
    save_ssl_confusion_matrices(run_path, model, loaders, device)


if __name__ == "__main__":
    main()
