"""
Evaluate a trained Building-Diffusion run on the test set (TPR similarity, FAR distance, FGW distance).
    python scripts/evaluate.py --run runs/<run_id> [--config configs/evaluate.yaml] [section.key=value ...]
"""
import os
import time
from datetime import datetime

import torch

from building_diffusion.config import EvaluateConfig, config_parser, load_config
from building_diffusion.constants import NUM_VOXEL_CLASSES
from building_diffusion.data.dataset import make_loader, make_splits
from building_diffusion.evaluation import (
    complete_test_evaluation,
    load_run_model_config,
    load_run_ssl_model_config,
    write_model_details,
)
from building_diffusion.models.diffusion import build_diffusion_model
from building_diffusion.models.layers import position_encoder
from building_diffusion.models.noise_scheduler import NoiseScheduler
from building_diffusion.utils.seed import set_seed


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = config_parser("Evaluate a trained Building-Diffusion run.", "configs/evaluate.yaml")
    parser.add_argument("--run", required=True, help="Run folder, e.g. runs/dm_Sep28_12-04-04_GAT_256_3_500_1000")
    args = parser.parse_args()
    cfg = load_config(EvaluateConfig, args.config, args.overrides)

    run_path = os.path.normpath(args.run)
    model_cfg = load_run_model_config(run_path, cfg.eval.layer_name)
    ssl_model_cfg = load_run_ssl_model_config(run_path, model_cfg)

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = os.path.join(run_path, f"evaluation_{current_time}.log")
    print(f"Filename: {log_path}")
    output_file = open(log_path, mode='w')
    output_file.write("Building-Diffusion Evaluation Report\n")
    output_file.write("="*25)
    output_file.write(f"\nModel Foldername: {os.path.basename(run_path)}\n")
    output_file.write(f"Layer Name: {model_cfg.layer_name}\n")
    output_file.write(f"Number of Samples: {cfg.eval.num_samples}\n")
    output_file.write(f"Seed: {cfg.eval.seed} (deterministic: {cfg.eval.deterministic})\n")
    output_file.write(f"Data & Time: {datetime.now()}\n")

    # The seed is set before the data loader and the model are created, as both draw random numbers
    set_seed(cfg.eval.seed, cfg.eval.deterministic)

    # Test set, rounded to a multiple of split_batch_size (32, the batch size used during training)
    data_fname_list = sorted(os.listdir(cfg.data.processed_dir))
    _, _, test_filenames = make_splits(data_fname_list, cfg.data.train_size, cfg.data.test_size, batch_size=cfg.eval.split_batch_size)
    test_data_loader = make_loader(cfg.data.processed_dir, test_filenames, batch_size=1, shuffle=False, num_workers=4)

    output_file.write(f"Test Size: {len(test_filenames)}\n")
    output_file.write("="*25)

    # Model
    noise_scheduler = NoiseScheduler(timesteps=model_cfg.diffusion_steps, x_classes=NUM_VOXEL_CLASSES, device=device)
    positional_encoding = position_encoder(device, hidden_dim=model_cfg.hidden_channels[0])

    # No SSL checkpoint: the diffusion checkpoint already contains the SSL weights
    diffusion_model = build_diffusion_model(model_cfg, device, ssl_model_cfg=ssl_model_cfg)
    checkpoint = os.path.join(run_path, "checkpoints", "best_diffusion_model.pt")
    diffusion_model.load_state_dict(torch.load(checkpoint, map_location=device))
    diffusion_model.eval()
    diffusion_model.to(device)
    write_model_details(output_file, model_cfg, NUM_VOXEL_CLASSES)

    # Evaluation
    eval_start = time.time()
    tpr_similarity, far_distance, normalized_fgw_distance = complete_test_evaluation(test_data_loader, diffusion_model, device,
                                                                                      noise_scheduler=noise_scheduler,
                                                                                      timesteps=model_cfg.diffusion_steps,
                                                                                      positional_encoding=positional_encoding,
                                                                                      raw_dir=cfg.data.raw_dir,
                                                                                      num_inferences=cfg.eval.num_samples)

    output_file.write(f"\nEvaluation Results Using {cfg.eval.num_samples} Inferences\n")
    output_file.write(f"Test TPR Loss (max.): {tpr_similarity:.4f} | Test FAR Loss (min.): {far_distance:.4f} | FGW Distance (min.): {normalized_fgw_distance:.4f}\n")

    output_file.write("="*25)
    eval_end = time.time()
    output_file.write(f"\nComplete Evaluation Duration: {time.strftime('%H:%M:%S', time.gmtime(eval_end - eval_start))}\n")
    output_file.close()


if __name__ == "__main__":
    main()
