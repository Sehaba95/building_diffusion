import os

from tensorboardX import SummaryWriter

from building_diffusion.config import save_config


def create_run(run_id, config, run_dir="runs"):
    """
    Create runs/<run_id>/ and its checkpoints/ folder, save the resolved config to config.yaml,
    and open a TensorBoard writer and the run log file.
    Returns (run_path, checkpoint_dir, writer, log_file).
    """
    run_path = os.path.join(run_dir, run_id)
    model_dir = os.path.join(run_path, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    for directory in (run_path, model_dir):
        os.mkdir(directory)

    save_config(config, os.path.join(run_path, "config.yaml"))

    writer = SummaryWriter(log_dir=run_path)
    log_file = open(os.path.join(run_path, "model_{}.log".format(run_id)), mode='x')
    return run_path, model_dir, writer, log_file
