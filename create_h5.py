import json
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================================
# ============================== Configuration =============================
# =========================================================================

# Root directory containing all MATLAB output folders.
INPUT_BASE_FOLDER = "./min"

# Output directory for the generated HDF5 files.
OUTPUT_HDF5_FOLDER = "./processed_data1001"

# Node metadata file used to determine matrix ordering and dimensions.
NODE_INFO_FILE = "./node_info.json"

# =========================================================================


def build_h5_file(h5_path, vol_folders, covol_folders, node_order):
    """
    Build an HDF5 file from volatility and co-volatility CSV inputs.
    """
    num_nodes = len(node_order)
    node_to_idx = {name: i for i, name in enumerate(node_order)}

    print(f"\n--- Loading data for {os.path.basename(h5_path)} ---")

    # --- 1. Automatically determine the reference number of days ---
    final_days_length = None
    for folder in vol_folders:
        if os.path.exists(folder) and os.listdir(folder):
            first_file_path = os.path.join(folder, os.listdir(folder)[0])
            df_first = pd.read_csv(first_file_path, header=None)
            final_days_length = df_first.shape[1]
            print(
                f"[Debug] Automatically detected reference day count from "
                f"'{os.path.basename(first_file_path)}': {final_days_length}"
            )
            break

    if final_days_length is None:
        print("Error: all volatility folders are empty, so the reference day count could not be determined.")
        return

    # a. Load all volatility data.
    vol_data = {}
    for folder in vol_folders:
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                asset_id = os.path.splitext(file)[0]
                if asset_id in node_order:
                    df = pd.read_csv(os.path.join(folder, file), header=None)
                    if df.shape[1] >= final_days_length:
                        vol_data[asset_id] = df.iloc[:, -final_days_length:].values
                    else:
                        print(
                            f"  -> Warning: volatility file {file} has only {df.shape[1]} days, "
                            f"which is less than the required {final_days_length}. Skipped."
                        )

    # b. Load all co-volatility data.
    covol_data = {}
    for group, folder in covol_folders.items():
        if not os.path.exists(folder):
            continue
        print(f"Loading group: {group}")
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                pair_name = os.path.splitext(file)[0]
                suffixes_to_remove = ["_covol_of_vol", "_covol", "_vol_of_vol", "_vol"]
                for suffix in suffixes_to_remove:
                    if pair_name.endswith(suffix):
                        pair_name = pair_name[: -len(suffix)]
                        break

                df = pd.read_csv(os.path.join(folder, file), header=None)
                if df.shape[1] >= final_days_length:
                    covol_data[pair_name] = df.iloc[:, -final_days_length:].values
                else:
                    print(
                        f"  -> Warning: co-volatility file {file} has only {df.shape[1]} days, "
                        f"which is less than the required {final_days_length}. Skipped."
                    )

    # --- 2. Determine time and dimensionality ---
    if not vol_data:
        print("Error: no valid volatility data were loaded, so processing cannot continue.")
        return

    num_timesteps = next(iter(vol_data.values())).shape[1]
    num_intraday_points = next(iter(vol_data.values())).shape[0]

    print(
        f"Data loading finished. All series were truncated to {num_timesteps} days "
        f"with {num_intraday_points} intraday points per day."
    )

    # --- 3. Build matrices timestep by timestep and write to HDF5 ---
    with h5py.File(h5_path, "w") as f:
        global_timestep_idx = 0
        desc = f"Building matrices ({os.path.basename(h5_path)})"
        for day_idx in tqdm(range(num_timesteps), desc=desc):
            for point_idx in range(num_intraday_points):
                matrix = np.zeros((num_nodes, num_nodes))

                # a. Fill the diagonal with volatilities.
                for asset_id, data in vol_data.items():
                    if asset_id in node_to_idx:
                        idx = node_to_idx[asset_id]
                        matrix[idx, idx] = data[point_idx, day_idx]

                # b. Fill the off-diagonal entries with co-volatilities.
                for pair_name, data in covol_data.items():
                    try:
                        id1, id2 = pair_name.split("_")
                        if id1 in node_to_idx and id2 in node_to_idx:
                            idx1 = node_to_idx[id1]
                            idx2 = node_to_idx[id2]
                            value = data[point_idx, day_idx]
                            matrix[idx1, idx2] = value
                            matrix[idx2, idx1] = value
                    except (ValueError, KeyError):
                        continue

                # c. Write the matrix for the current timestep to HDF5.
                f.create_dataset(str(global_timestep_idx), data=matrix, dtype=np.float64)
                global_timestep_idx += 1

    print(f"Successfully created HDF5 file: {os.path.basename(h5_path)}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_HDF5_FOLDER):
        os.makedirs(OUTPUT_HDF5_FOLDER)

    try:
        with open(NODE_INFO_FILE, "r", encoding="utf-8") as f:
            node_info = json.load(f)
        NODE_ORDER = node_info["node_order"]
    except FileNotFoundError:
        raise Exception(f"Error: node metadata file {NODE_INFO_FILE} was not found.")

    # --- Task 1: build vol_covol.h5 ---
    vol_covol_h5_path = os.path.join(OUTPUT_HDF5_FOLDER, "vol_covol.h5")
    vol_folders_1 = [
        os.path.join(INPUT_BASE_FOLDER, "stock_vol"),
        os.path.join(INPUT_BASE_FOLDER, "energy_vol"),
    ]
    covol_folders_1 = {
        "stock_stock": os.path.join(INPUT_BASE_FOLDER, "covol_stock_stock"),
        "stock_energy": os.path.join(INPUT_BASE_FOLDER, "covol_stock_energy"),
    }
    build_h5_file(vol_covol_h5_path, vol_folders_1, covol_folders_1, NODE_ORDER)

    # --- Task 2: build volvol_covolvol.h5 ---
    volvol_covolvol_h5_path = os.path.join(OUTPUT_HDF5_FOLDER, "volvol_covolvol.h5")
    vol_folders_2 = [
        os.path.join(INPUT_BASE_FOLDER, "stock_vol_of_vol"),
        os.path.join(INPUT_BASE_FOLDER, "energy_vol_of_vol"),
    ]
    covol_folders_2 = {
        "stock_stock": os.path.join(INPUT_BASE_FOLDER, "covol_of_vol_stock_stock"),
        "stock_energy": os.path.join(INPUT_BASE_FOLDER, "covol_of_vol_stock_energy"),
    }
    build_h5_file(volvol_covolvol_h5_path, vol_folders_2, covol_folders_2, NODE_ORDER)

    print("\n--- All HDF5 files have been created. ---")
