import json
import math
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils.datasetchange import FinalHeteroDataset
from utils.modelschange import HeteroGNNModel


def train(p: dict):
    """
    Run the full training, validation, and testing workflow.
    """
    # --- 1. Set paths and random seeds ---
    folder_path = f"output/{p['modelname']}_{p['seq_length']}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f"{folder_path}/GNN_param_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(p, f)

    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    # --- 2. Initialize the heterogeneous dataset ---
    print("--- Initializing the heterogeneous dataset (FinalHeteroDataset)... ---")
    dataset = FinalHeteroDataset(
        hdf5_file_vol=p["hdf5_file_vol_std"],
        hdf5_file_volvol=p["hdf5_file_volvol_std"],
        stock_har_rv_folder=p["stock_har_rv_folder"],
        energy_har_rv_folder=p["energy_har_rv_folder"],
        stock_energy_corr_folder=p["stock_energy_corr_folder"],
        energy_energy_corr_folder=p["energy_energy_corr_folder"],
        node_info_file=p["node_info_file"],
        root=f"processed_data1001/final_hetero_dataset_seq_{p['seq_length']}",
        seq_length=p["seq_length"],
        intraday_points=p["intraday_points"],
    )

    # --- 3. Split into train, validation, and test sets ---
    train_size = int(p["train_proportion"] * len(dataset))
    validation_size = int(p["validation_proportion"] * len(dataset))

    train_dataset = dataset[:train_size]
    validation_dataset = dataset[train_size: train_size + validation_size]
    test_dataset = dataset[train_size + validation_size:]

    print(
        f"--- Dataset split completed: {len(train_dataset)} (train), "
        f"{len(validation_dataset)} (validation), {len(test_dataset)} (test) ---"
    )

    train_loader = DataLoader(train_dataset, batch_size=p["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=p["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=p["batch_size"], shuffle=False)

    # --- 4. Load required metadata ---
    with open(p["node_info_file"], "r", encoding="utf-8") as f:
        node_info = json.load(f)
    num_stocks = len(node_info["stock_ids"])
    num_energy = len(node_info["energy_ids"])

    stock_feature_dim = (3 + 1 + (num_stocks - 1) + num_energy) * p["seq_length"]
    energy_feature_dim = (3 + 1 + num_stocks + (num_energy - 1)) * p["seq_length"]
    edge_feature_dim = 3 * p["seq_length"]

    # --- 5. Initialize the model ---
    print("--- Initializing the heterogeneous GNN model (HeteroGNNModel)... ---")
    model = HeteroGNNModel(
        stock_feature_dim=stock_feature_dim,
        energy_feature_dim=energy_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_channels=p["hidden_channels"],
        heads=p["num_heads"],
        num_layers=p["num_layers"],
        high_freq_output_dim=num_stocks,
        low_freq_output_dim=num_stocks,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- 6. Set optimizer and loss function ---
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = torch.nn.MSELoss()

    # --- 7. Training and validation loop ---
    print("\n--- Starting training and validation ---")
    min_validation_loss = float("inf")
    best_epoch = -1

    for epoch in range(p["num_epochs"]):
        # --- Training phase ---
        model.train()
        total_train_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{p['num_epochs']} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            y_high, y_low = data["stock"].y_high, data["stock"].y_low
            loss_high = criterion(pred_high, y_high)
            loss_low = criterion(pred_low, y_low)
            loss = p["lambda_weight"] * loss_low + (1 - p["lambda_weight"]) * loss_high
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        total_validation_loss = 0
        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                y_high, y_low = data["stock"].y_high, data["stock"].y_low
                loss_high = criterion(pred_high, y_high)
                loss_low = criterion(pred_low, y_low)
                loss = p["lambda_weight"] * loss_low + (1 - p["lambda_weight"]) * loss_high
                total_validation_loss += loss.item()
        avg_validation_loss = total_validation_loss / len(validation_loader)

        print(
            f"Epoch {epoch + 1}/{p['num_epochs']} | "
            f"Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_validation_loss:.6f}"
        )

        # --- Save the best model ---
        if avg_validation_loss < min_validation_loss:
            min_validation_loss = avg_validation_loss
            best_epoch = epoch + 1
            save_path = os.path.join(folder_path, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Validation loss decreased. Model saved to {save_path}")

    print(
        f"\n--- Training completed. Best model found at epoch {best_epoch} "
        f"(Validation Loss: {min_validation_loss:.6f}) ---"
    )

    # --- 8. Final test evaluation using the best saved model ---
    print("\n--- Starting final evaluation on the test set... ---")

    # Load statistics used for de-standardization.
    with open("./processed_data1001/vol_covol_stats.json", "r", encoding="utf-8") as f:
        vol_stats = json.load(f)
    diag_mean_high = vol_stats["stock_diag_mean"]
    diag_std_high = vol_stats["stock_diag_mean"]

    with open("./processed_data1001/har_rv_stats.json", "r", encoding="utf-8") as f:
        rv_stats = json.load(f)
    diag_mean_low = rv_stats["daily_mean"]
    diag_std_low = rv_stats["daily_std"]

    # Load the best model weights.
    best_model_path = os.path.join(folder_path, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Warning: best model checkpoint not found, so final testing cannot proceed.")
        return

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    total_unstd_mse_high = 0
    total_unstd_mse_low = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="[Final Test]"):
            data = data.to(device)
            pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            y_high, y_low = data["stock"].y_high, data["stock"].y_low

            # De-standardize predictions and targets.
            pred_high_unstd = (pred_high * diag_std_high) + diag_mean_high
            y_high_unstd = (y_high * diag_std_high) + diag_mean_high
            pred_low_unstd = (pred_low * diag_std_low) + diag_mean_low
            y_low_unstd = (y_low * diag_std_low) + diag_mean_low

            # Accumulate MSE on the original scale.
            total_unstd_mse_high += criterion(pred_high_unstd, y_high_unstd).item() * y_high.size(0)
            total_unstd_mse_low += criterion(pred_low_unstd, y_low_unstd).item() * y_low.size(0)

    # Compute final MSE and RMSE.
    avg_unstd_mse_high = total_unstd_mse_high / len(test_dataset)
    avg_unstd_mse_low = total_unstd_mse_low / len(test_dataset)
    final_rmse_high = math.sqrt(avg_unstd_mse_high)
    final_rmse_low = math.sqrt(avg_unstd_mse_low)

    print("\n--- Final test results ---")
    print(f"  - High-frequency RMSE (y_high): {final_rmse_high:.12f}")
    print(f"  - High-frequency MSE (y_high):  {avg_unstd_mse_high:.12f}")
    print(f"  - Low-frequency RMSE (y_low):   {final_rmse_low:.12f}")
    print(f"  - Low-frequency MSE (y_low):    {avg_unstd_mse_low:.12f}")


if __name__ == "__main__":
    with open("config/GNN_param_optuna.yaml", "r", encoding="utf-8") as f:
        p = yaml.safe_load(f)

    # Make sure the config file contains split proportions.
    if "train_proportion" not in p or "validation_proportion" not in p:
        raise ValueError(
            "The config file 'GNN_param.yaml' must contain both "
            "'train_proportion' and 'validation_proportion'."
        )

    train(p=p)
