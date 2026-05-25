import os

import pandas as pd  # Imported because FinalHeteroDataset depends on pandas internally.
import yaml

from utils.datasetchange import FinalHeteroDataset


def create_all_caches():
    """
    Batch-create dataset caches for every seq_length listed in the tuning config.

    The script reads the Optuna configuration file and creates a dedicated
    PyTorch Geometric cache directory for each configured sequence length.
    """
    # --- 1. Load the hyperparameter tuning config file ---
    try:
        config_path = "./config/GNN_param_optuna.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            p = yaml.safe_load(f)
        print(f"--- Successfully loaded config file: {config_path} ---")
    except FileNotFoundError:
        print(f"Error: tuning config file '{config_path}' was not found.")
        return

    # --- 2. Get all seq_length values that need to be processed ---
    try:
        seq_lengths_to_process = p["hyperparameters"]["seq_length"][0]
        if not isinstance(seq_lengths_to_process, list):
            print("Error: the seq_length entry in the config file has an invalid format.")
            return
        print(f"--- Preparing caches for the following seq_length values: {seq_lengths_to_process} ---")
    except (KeyError, IndexError):
        print("Error: could not find the 'seq_length' setting in the config file.")
        return

    # --- 3. Create each cache in a loop ---
    for seq_length in seq_lengths_to_process:
        print("\n=================================================")
        print(f"====== Start processing seq_length = {seq_length} ======")
        print("=================================================")

        # Define the cache path dedicated to the current seq_length.
        root_path = f"processed_data1001/final_hetero_dataset_seq_{seq_length}"

        # Skip if the cache already exists.
        if os.path.exists(os.path.join(root_path, "processed", "data.pt")):
            print(f"--- Cache already exists at '{root_path}'. Skipping. ---")
            continue

        try:
            # Instantiating the dataset triggers process() automatically when
            # the cache is missing, which creates the processed dataset files.
            print(f"--- Creating a new dataset cache for seq_length={seq_length}... ---")
            FinalHeteroDataset(
                hdf5_file_vol=p["hdf5_file_vol_std"],
                hdf5_file_volvol=p["hdf5_file_volvol_std"],
                stock_har_rv_folder=p["stock_har_rv_folder"],
                energy_har_rv_folder=p["energy_har_rv_folder"],
                stock_energy_corr_folder=p["stock_energy_corr_folder"],
                energy_energy_corr_folder=p["energy_energy_corr_folder"],
                node_info_file=p["node_info_file"],
                root=root_path,
                seq_length=seq_length,
                intraday_points=p["intraday_points"],
            )
            print(f"--- Cache for seq_length = {seq_length} created successfully. ---")

        except Exception as e:
            print(f"!!!!!! A critical error occurred while processing seq_length = {seq_length}: {e} !!!!!!")
            # Continue with the next candidate even if one run fails.
            continue

    print("\n=================================================")
    print("====== All requested cache files have been processed. ======")
    print("=================================================")


if __name__ == "__main__":
    # Make sure utils.datasetchange is the offline version whose process()
    # method contains the full feature concatenation logic.
    create_all_caches()
