"""
Convert the raw Building-GAN dataset (JSON files) into one processed graph per building (.pt).

Raw folder structure:
raw_dir
    - global_graph_data
    - local_graph_data
    - voxel_data

    python scripts/preprocess.py [--raw_dir Data/6types-raw_data/6types-raw_data] [--out_dir Data/6types-processed_data]
"""
import argparse
import os

import torch
from tqdm import tqdm

from building_diffusion.data.constructor import GraphConstructor


def main():
    parser = argparse.ArgumentParser(description="Process the raw Building-GAN dataset into graphs.")
    parser.add_argument("--raw_dir", default="Data/6types-raw_data/6types-raw_data", help="Raw dataset folder")
    parser.add_argument("--out_dir", default="Data/6types-processed_data", help="Output folder for the processed graphs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)  # exist_ok=True avoids error if folder exists
    global_graph_dir = os.path.join(args.raw_dir, "global_graph_data")

    for fname in tqdm(sorted(os.listdir(global_graph_dir))):
        data_id = int(''.join(filter(str.isdigit, fname)))

        g = GraphConstructor.load_graph_jsons(data_id, args.raw_dir)
        output_fname = "data" + str(data_id).zfill(GraphConstructor.data_id_length) + ".pt"
        torch.save(g, os.path.join(args.out_dir, output_fname))

    print("Data processing completed")


if __name__ == "__main__":
    main()
