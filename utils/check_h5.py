import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from datasetchange import FinalHomogeneousDataset


def verify_dataset_features():
    """
    Initialize the dataset, extract one sample, and save the node features for
    one chosen stock and one chosen energy asset to separate CSV files.
    """
    print("\n\n=============================================")
    print("========= Starting dataset verification =========")
    print("=============================================")

    # --- 1. Configuration ---
    HDF5_VOL_PATH = "./processed_data/vol_covol_standardized.h5"
    HDF5_VOLVOL_PATH = "./processed_data/volvol_covolvol_standardized.h5"
    STOCK_HAR_FOLDER = "./day/daily_stock_har"
    ENERGY_HAR_FOLDER = "./day/daily_energy_har"
    STOCK_ENERGY_CORR_FOLDER = "./day/daycorr_stock_energy"
    ENERGY_ENERGY_CORR_FOLDER = "./day/daycorr_energy_energy"
    NODE_INFO_FILE = "./node_info.json"
    SEQ_LENGTH = 21
    INTRADAY_POINTS = 3

    # Choose the asset IDs to inspect here.
    STOCK_TO_VERIFY = "000932华菱钢铁"  # Change to the stock ID you want to inspect.
    ENERGY_TO_VERIFY = "CU.SHF"  # Change to the energy ID you want to inspect.

    OUTPUT_STOCK_CSV = f"verified_stock_{STOCK_TO_VERIFY}_features.csv"
    OUTPUT_ENERGY_CSV = f"verified_energy_{ENERGY_TO_VERIFY}_features.csv"

    try:
        print("Initializing the dataset class...")
        dataset_root = "processed_data/final_homogeneous_dataset/"
        processed_file = os.path.join(dataset_root, "processed", "data.pt")
        if os.path.exists(processed_file):
            print(f"Old cache file '{processed_file}' found. Deleting it...")
            os.remove(processed_file)

        dataset = FinalHomogeneousDataset(
            hdf5_file_vol=HDF5_VOL_PATH,
            hdf5_file_volvol=HDF5_VOLVOL_PATH,
            stock_har_rv_folder=STOCK_HAR_FOLDER,
            energy_har_rv_folder=ENERGY_HAR_FOLDER,
            stock_energy_corr_folder=STOCK_ENERGY_CORR_FOLDER,
            energy_energy_corr_folder=ENERGY_ENERGY_CORR_FOLDER,
            node_info_file=NODE_INFO_FILE,
            seq_length=SEQ_LENGTH,
            intraday_points=INTRADAY_POINTS,
            root=dataset_root,
        )
        print("Dataset initialization and processing succeeded.")
    except Exception as e:
        print(f"Dataset initialization or processing failed: {e}")
        return

    if len(dataset) == 0:
        print("Error: the processed dataset is empty, so no sample can be extracted.")
        return

    print("Extracting the first training sample (dataset[0])...")
    data_sample = dataset[0]
    node_features_tensor = data_sample.x

    print(f"Extraction succeeded. Node feature matrix 'x' has shape: {node_features_tensor.shape}")
    print(f"Generating detailed feature reports for stock '{STOCK_TO_VERIFY}' and energy '{ENERGY_TO_VERIFY}'...")

    with open(NODE_INFO_FILE, "r", encoding="utf-8") as f:
        node_info = json.load(f)

    stock_ids = node_info["stock_ids"]
    energy_ids = node_info["energy_ids"]
    node_order = node_info["node_order"]

    # --- Process and save the selected stock features ---
    if STOCK_TO_VERIFY in node_order:
        stock_idx = node_order.index(STOCK_TO_VERIFY)
        base_names = []
        base_names.extend(["HAR_1", "HAR_2", "HAR_3"])
        base_names.append("Inst_Vol")
        base_names.extend([f"Covol_w_{sid}" for sid in stock_ids if sid != STOCK_TO_VERIFY])
        base_names.extend([f"Corr_w_{eid}" for eid in energy_ids])
        column_names = [f"{name}_lag_{i}" for i in range(SEQ_LENGTH) for name in base_names]

        if node_features_tensor.shape[1] == len(column_names):
            stock_series = pd.Series(node_features_tensor[stock_idx].numpy(), index=column_names, name=STOCK_TO_VERIFY)
            stock_series.to_csv(OUTPUT_STOCK_CSV, header=True)
            print(
                f"\nVerification succeeded. Features for stock '{STOCK_TO_VERIFY}' "
                f"were saved to '{os.path.abspath(OUTPUT_STOCK_CSV)}'"
            )
        else:
            print(f"\nWarning: stock '{STOCK_TO_VERIFY}' feature dimensions do not match. Report not generated.")
    else:
        print(f"\nError: stock ID '{STOCK_TO_VERIFY}' was not found in node_info.json.")

    # --- Process and save the selected energy features ---
    if ENERGY_TO_VERIFY in node_order:
        energy_idx = node_order.index(ENERGY_TO_VERIFY)
        base_names = []
        base_names.extend(["HAR_1", "HAR_2", "HAR_3"])
        base_names.append("Inst_Vol")
        base_names.extend([f"Covol_w_{sid}" for sid in stock_ids])
        base_names.extend([f"Corr_w_{eid}" for eid in energy_ids if eid != ENERGY_TO_VERIFY])
        column_names = [f"{name}_lag_{i}" for i in range(SEQ_LENGTH) for name in base_names]

        if node_features_tensor.shape[1] == len(column_names):
            energy_series = pd.Series(node_features_tensor[energy_idx].numpy(), index=column_names, name=ENERGY_TO_VERIFY)
            energy_series.to_csv(OUTPUT_ENERGY_CSV, header=True)
            print(
                f"Verification succeeded. Features for energy asset '{ENERGY_TO_VERIFY}' "
                f"were saved to '{os.path.abspath(OUTPUT_ENERGY_CSV)}'"
            )
        else:
            print(f"\nWarning: energy asset '{ENERGY_TO_VERIFY}' feature dimensions do not match. Report not generated.")
    else:
        print(f"\nError: energy ID '{ENERGY_TO_VERIFY}' was not found in node_info.json.")

    print("\nYou can now inspect the generated files in Excel.")


if __name__ == "__main__":
    verify_dataset_features()
