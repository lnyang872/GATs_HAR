import json
import os

import pandas as pd
import torch

# Important: make sure this imports the final dataset class implementation.
from datasetchange import FinalHomogeneousDataset


def verify_dataset_features():
    """
    Initialize the final dataset, extract one sample, and save its node
    features to a CSV file for inspection.
    """
    print("--- Starting dataset verification ---")

    # --- 1. Configuration ---
    # Update these paths to match your local data layout if needed.
    HDF5_VOL_PATH = "./processed_data/vol_covol.h5"
    HDF5_VOLVOL_PATH = "./processed_data/volvol_covolvol.h5"
    STOCK_HAR_FOLDER = "./day/daily_stock_har"
    ENERGY_HAR_FOLDER = "./day/daily_energy_har"

    # Use the two correlation folders separately.
    STOCK_ENERGY_CORR_FOLDER = "./day/daycorr_stock_energy"
    ENERGY_ENERGY_CORR_FOLDER = "./day/daycorr_energy_energy"

    NODE_INFO_FILE = "./node_info.json"
    SEQ_LENGTH = 22  # Sequence length used by the model.

    OUTPUT_CSV_FILENAME = "verified_node_features_final.csv"

    # --- 2. Initialize the dataset ---
    try:
        print("Initializing the dataset class...")
        dataset = FinalHomogeneousDataset(
            hdf5_file_vol=HDF5_VOL_PATH,
            hdf5_file_volvol=HDF5_VOLVOL_PATH,
            stock_har_rv_folder=STOCK_HAR_FOLDER,
            energy_har_rv_folder=ENERGY_HAR_FOLDER,
            stock_energy_corr_folder=STOCK_ENERGY_CORR_FOLDER,
            energy_energy_corr_folder=ENERGY_ENERGY_CORR_FOLDER,
            node_info_file=NODE_INFO_FILE,
            seq_length=SEQ_LENGTH,
        )
        print("Dataset initialization succeeded.")
    except Exception as e:
        print(f"Dataset initialization failed: {e}")
        print("Check the file paths above and make sure the cache matches the current code version.")
        return

    # --- 3. Extract the first sample ---
    if len(dataset) == 0:
        print("Error: the processed dataset is empty, so no sample can be extracted.")
        print("Please check the process() logic and data alignment.")
        return

    print("Extracting the first training sample (dataset[0])...")
    data_sample = dataset[0]
    node_features_tensor = data_sample.x

    print(f"Extraction succeeded. Node feature matrix 'x' has shape: {node_features_tensor.shape}")

    # --- 4. Convert the tensor to a labeled DataFrame ---
    print("Converting the feature matrix to a readable DataFrame...")

    with open(NODE_INFO_FILE, "r", encoding="utf-8") as f:
        node_info = json.load(f)
    node_names = node_info["node_order"]
    num_stocks = len(node_info["stock_ids"])
    num_energy = len(node_info["energy_ids"])

    # Based on the final feature design:
    # Stock dimension: 3(HAR) + 1(Vi') + (N-1)(Cij_stock) + M(Corr_energy) = N+M+3
    # Energy dimension: 3(HAR) + 1(Vi') + N(Cij_stock) + (M-1)(Corr_energy) = N+M+3
    total_base_features = num_stocks + num_energy + 3

    # Name the common part explicitly and use generic names for the rest.
    base_feature_names = ["HAR_1", "HAR_2", "HAR_3", "Inst_Vol"]
    remaining_dims = total_base_features - 4
    base_feature_names.extend([f"Relational_Feat_{i + 1}" for i in range(remaining_dims)])

    column_names = [f"{name}_lag_{i}" for i in range(SEQ_LENGTH) for name in base_feature_names]

    if node_features_tensor.shape[1] != len(column_names):
        print("\nWarning: the generated column-name count does not match the actual feature dimension.")
        print(f"Actual dimension: {node_features_tensor.shape[1]}, expected dimension: {len(column_names)}")
        print("Falling back to numeric column names. Please review the dataset feature concatenation logic.")
        column_names = [f"feature_{i + 1}" for i in range(node_features_tensor.shape[1])]

    features_df = pd.DataFrame(node_features_tensor.numpy(), index=node_names, columns=column_names)

    # --- 5. Save the CSV file ---
    try:
        features_df.to_csv(OUTPUT_CSV_FILENAME)
        print(f"\nVerification succeeded. The first sample has been saved to: '{os.path.abspath(OUTPUT_CSV_FILENAME)}'")
        print("You can now inspect this file in Excel.")
    except Exception as e:
        print(f"Error while saving the CSV file: {e}")


if __name__ == "__main__":
    verify_dataset_features()
