import json
import math
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler

from utils.datasetfinal import FinalHeteroDataset
from utils.feature_config import (
    build_full_dataset_cache_name,
    get_feature_dimensions,
    get_training_objective,
    sync_feature_flags,
    validate_feature_dimensions,
)
from utils.losses import QLIKELoss
from utils.modelschange import HeteroGNNModel


class DayGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset_length: int,
        intraday_points: int,
        days_per_batch: int,
        shuffle_days: bool,
    ):
        if intraday_points <= 0:
            raise ValueError("intraday_points must be positive.")
        if days_per_batch <= 0:
            raise ValueError("days_per_batch must be positive.")
        if dataset_length % intraday_points != 0:
            raise ValueError(
                "Dataset length must be divisible by intraday_points "
                "for day-grouped batching."
            )

        self.dataset_length = dataset_length
        self.intraday_points = intraday_points
        self.days_per_batch = days_per_batch
        self.shuffle_days = shuffle_days
        self.num_days = dataset_length // intraday_points

    def __iter__(self):
        day_indices = np.arange(self.num_days)
        if self.shuffle_days:
            np.random.shuffle(day_indices)

        for start in range(0, self.num_days, self.days_per_batch):
            batch_days = day_indices[start : start + self.days_per_batch]
            batch_indices = []
            for day_idx in batch_days:
                day_start = int(day_idx) * self.intraday_points
                batch_indices.extend(
                    range(day_start, day_start + self.intraday_points)
                )
            yield batch_indices

    def __len__(self):
        return math.ceil(self.num_days / self.days_per_batch)


def _build_dataset(p: dict) -> FinalHeteroDataset:
    include_energy = p.get("include_energy", True)
    node_flags, edge_flags = sync_feature_flags(p)
    dataset_root_name = build_full_dataset_cache_name(p)

    return FinalHeteroDataset(
        hdf5_file_vol=p["hdf5_file_vol_std"],
        hdf5_file_volvol=p["hdf5_file_volvol_std"],
        stock_har_rv_folder=p["stock_har_rv_folder"],
        energy_har_rv_folder=p["energy_har_rv_folder"],
        stock_energy_corr_folder=p["stock_energy_corr_folder"],
        energy_energy_corr_folder=p["energy_energy_corr_folder"],
        node_info_file=p["node_info_file"],
        root=f"processed_data1001/{dataset_root_name}",
        seq_length=p["seq_length"],
        intraday_points=p["intraday_points"],
        include_energy=include_energy,
        use_har_features=node_flags["har"],
        node_feature_flags=node_flags,
        edge_feature_flags=edge_flags,
    )


def _get_model_dimensions(p: dict, node_info: dict) -> dict:
    include_energy = p.get("include_energy", True)
    node_flags, edge_flags = sync_feature_flags(p)
    dimensions = get_feature_dimensions(
        num_stocks=len(node_info["stock_ids"]),
        num_energy=len(node_info["energy_ids"]) if include_energy else 0,
        seq_length=p["seq_length"],
        include_energy=include_energy,
        node_flags=node_flags,
        edge_flags=edge_flags,
    )
    validate_feature_dimensions(dimensions, include_energy)
    return dimensions


def _build_model(p: dict, node_info: dict) -> HeteroGNNModel:
    dimensions = _get_model_dimensions(p, node_info)
    num_stocks = len(node_info["stock_ids"])
    include_energy = p.get("include_energy", True)
    node_flags, _ = sync_feature_flags(p)

    return HeteroGNNModel(
        stock_feature_dim=dimensions["stock_feature_dim"],
        energy_feature_dim=dimensions["energy_feature_dim"],
        edge_feature_dim=dimensions["edge_feature_dim"],
        heads=p["num_heads"],
        high_freq_output_dim=num_stocks,
        low_freq_output_dim=num_stocks,
        hidden_layout=p["hidden_layout"],
        dropout=p.get("dropout", 0.0),
        activation=p.get("activation", "relu"),
        include_energy=include_energy,
        use_har_features=node_flags["har"],
    )


def _get_optimizer(model: torch.nn.Module, p: dict):
    optimizer_name = p["optimizer"]
    learning_rate = p["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _get_loss_function(p: dict):
    loss_name = p["loss_function"]
    if loss_name == "mse":
        return torch.nn.MSELoss(reduction="none")
    if loss_name == "qlike":
        return QLIKELoss(reduction="none")
    raise ValueError(f"Unsupported loss function: {loss_name}")


def _collect_edge_attr_dict(data) -> dict:
    edge_attr_dict = {}
    for rel in data.edge_types:
        if "edge_attr" in data[rel]:
            edge_attr_dict[rel] = data[rel].edge_attr
    return edge_attr_dict


def _get_days_per_batch(batch_size: int, intraday_points: int) -> int:
    if batch_size < intraday_points:
        raise ValueError(
            f"batch_size={batch_size} is smaller than intraday_points={intraday_points}."
        )
    return max(1, batch_size // intraday_points)


def _build_loader(dataset, *, intraday_points: int, batch_size: int, shuffle_days: bool, num_workers: int):
    days_per_batch = _get_days_per_batch(batch_size, intraday_points)
    effective_batch_size = days_per_batch * intraday_points

    if effective_batch_size != batch_size:
        print(
            f"--- [Batching] requested batch_size={batch_size}, "
            f"using grouped batch_size={effective_batch_size} "
            f"({days_per_batch} days x {intraday_points} intraday points)"
        )

    batch_sampler = DayGroupedBatchSampler(
        dataset_length=len(dataset),
        intraday_points=intraday_points,
        days_per_batch=days_per_batch,
        shuffle_days=shuffle_days,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
    )


def _aggregate_low_frequency_by_day(values: torch.Tensor, intraday_points: int) -> torch.Tensor:
    if values.numel() == 0:
        return values
    if values.size(0) % intraday_points != 0:
        raise ValueError(
            "Batch does not align with complete day groups. "
            "Check grouped batch sampler settings."
        )
    num_days = values.size(0) // intraday_points
    return values.reshape(num_days, intraday_points, -1).mean(dim=1)


def _compute_training_loss(
    pred_high,
    pred_low,
    y_high,
    y_low,
    criterion,
    p: dict,
):
    intraday_points = p.get("intraday_points", 3)
    loss_high = criterion(pred_high, y_high).mean(dim=0).mean()
    pred_low_daily = _aggregate_low_frequency_by_day(pred_low, intraday_points)
    y_low_daily = _aggregate_low_frequency_by_day(y_low, intraday_points)
    loss_low = criterion(pred_low_daily, y_low_daily).mean(dim=0).mean()

    objective = get_training_objective(p)
    if objective == "joint":
        lambda_weight = p.get("lambda_weight", 0.5)
        total_loss = lambda_weight * loss_low + (1 - lambda_weight) * loss_high
    elif objective == "high_only":
        total_loss = loss_high
    else:
        total_loss = loss_low

    return total_loss, loss_high, loss_low


def train(p: dict, trial_folder: str) -> dict:
    os.makedirs(trial_folder, exist_ok=True)
    with open(os.path.join(trial_folder, "GNN_param_used.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(p, f, allow_unicode=True)

    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    include_energy = p.get("include_energy", True)
    node_flags, edge_flags = sync_feature_flags(p)
    training_objective = get_training_objective(p)

    print(f"--- [Config] include_energy = {include_energy}")
    print(f"--- [Config] node_feature_flags = {node_flags}")
    print(f"--- [Config] edge_feature_flags = {edge_flags}")
    print(f"--- [Config] training_objective = {training_objective}")

    dataset = _build_dataset(p)

    intraday_points = p.get("intraday_points", 3)
    total_days = len(dataset) // intraday_points
    robustness_proportion = p.get("robustness_proportion", 0.7)
    robustness_limit_days = int(robustness_proportion * total_days)
    dataset = dataset[: robustness_limit_days * intraday_points]
    total_days = robustness_limit_days
    print(
        f"--- [Robustness] total days limited to {total_days} "
        f"({robustness_proportion * 100:.0f}%)"
    )

    train_days = int(p["train_proportion"] * total_days)
    validation_days = int(p["validation_proportion"] * total_days)
    train_size = train_days * intraday_points
    validation_end_idx = (train_days + validation_days) * intraday_points

    train_dataset = dataset[:train_size]
    validation_dataset = dataset[train_size:validation_end_idx]
    test_dataset = dataset[validation_end_idx:]

    train_loader = _build_loader(
        train_dataset,
        intraday_points=intraday_points,
        batch_size=p["batch_size"],
        shuffle_days=True,
        num_workers=p.get("num_workers", 4),
    )
    validation_loader = _build_loader(
        validation_dataset,
        intraday_points=intraday_points,
        batch_size=p["batch_size"],
        shuffle_days=False,
        num_workers=p.get("num_workers", 4),
    )
    test_loader = _build_loader(
        test_dataset,
        intraday_points=intraday_points,
        batch_size=p["batch_size"],
        shuffle_days=False,
        num_workers=p.get("num_workers", 4),
    )

    with open(p["node_info_file"], "r", encoding="utf-8") as f:
        node_info = json.load(f)

    model = _build_model(p, node_info)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = _get_optimizer(model, p)
    train_criterion = _get_loss_function(p)

    with open("./processed_data1001/vol_covol_stats.json", "r", encoding="utf-8") as f:
        vol_stats = json.load(f)
    diag_mean_high = vol_stats["stock_diag_mean"]
    diag_std_high = vol_stats["stock_diag_std"]

    with open("./processed_data1001/har_rv_stats.json", "r", encoding="utf-8") as f:
        rv_stats = json.load(f)
    diag_mean_low = rv_stats["daily_mean"]
    diag_std_low = rv_stats["daily_std"]

    min_validation_loss = float("inf")
    best_epoch = -1

    for epoch in range(p["num_epochs"]):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            edge_attr_dict = _collect_edge_attr_dict(data)

            pred_high, pred_low = model(
                data.x_dict, data.edge_index_dict, edge_attr_dict
            )
            if pred_high.numel() == 0:
                continue

            y_high = data["stock"].y_high
            y_low = data["stock"].y_low
            loss, _, _ = _compute_training_loss(
                pred_high=pred_high,
                pred_low=pred_low,
                y_high=y_high,
                y_low=y_low,
                criterion=train_criterion,
                p=p,
            )
            loss.backward()
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        validation_batches = 0

        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                edge_attr_dict = _collect_edge_attr_dict(data)
                pred_high, pred_low = model(
                    data.x_dict, data.edge_index_dict, edge_attr_dict
                )
                if pred_high.numel() == 0:
                    continue

                y_high = data["stock"].y_high
                y_low = data["stock"].y_low
                loss, _, _ = _compute_training_loss(
                    pred_high=pred_high,
                    pred_low=pred_low,
                    y_high=y_high,
                    y_low=y_low,
                    criterion=train_criterion,
                    p=p,
                )
                validation_loss += loss.item()
                validation_batches += 1

        avg_validation_loss = (
            validation_loss / validation_batches
            if validation_batches > 0
            else float("inf")
        )

        if avg_validation_loss < min_validation_loss:
            min_validation_loss = avg_validation_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(trial_folder, "best_model.pth"))

    best_model_path = os.path.join(trial_folder, "best_model.pth")
    if not os.path.exists(best_model_path):
        print(f"Warning: best model was not saved: {best_model_path}")
        return {"min_validation_loss": float("inf")}

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    num_stocks = len(node_info["stock_ids"])

    def evaluate_on_set(loader, dataset_ref):
        if not loader or len(dataset_ref) == 0:
            return {
                "mse_h": np.nan,
                "rmse_h": np.nan,
                "qlike_h": np.nan,
                "mse_l": np.nan,
                "rmse_l": np.nan,
                "qlike_l": np.nan,
            }

        mse_criterion = torch.nn.MSELoss(reduction="none")
        qlike_criterion = QLIKELoss(reduction="none")
        all_preds_h, all_actuals_h, all_preds_l, all_actuals_l = [], [], [], []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                edge_attr_dict = _collect_edge_attr_dict(data)
                pred_high, pred_low = model(
                    data.x_dict, data.edge_index_dict, edge_attr_dict
                )
                if pred_high.numel() == 0:
                    continue

                y_high = data["stock"].y_high
                y_low = data["stock"].y_low

                pred_high_unstd = (
                    pred_high * torch.tensor(diag_std_high, device=device)
                ) + torch.tensor(diag_mean_high, device=device)
                y_high_unstd = (
                    y_high * torch.tensor(diag_std_high, device=device)
                ) + torch.tensor(diag_mean_high, device=device)

                pred_low_unstd = (
                    pred_low * torch.tensor(diag_std_low, device=device)
                ) + torch.tensor(diag_mean_low, device=device)
                y_low_unstd = (
                    y_low * torch.tensor(diag_std_low, device=device)
                ) + torch.tensor(diag_mean_low, device=device)

                all_preds_h.append(pred_high_unstd)
                all_actuals_h.append(y_high_unstd)
                all_preds_l.append(pred_low_unstd)
                all_actuals_l.append(y_low_unstd)

        if not all_preds_h:
            return {
                "mse_h": np.nan,
                "rmse_h": np.nan,
                "qlike_h": np.nan,
                "mse_l": np.nan,
                "rmse_l": np.nan,
                "qlike_l": np.nan,
            }

        preds_h = torch.cat(all_preds_h, dim=0)
        actuals_h = torch.cat(all_actuals_h, dim=0)
        preds_l = torch.cat(all_preds_l, dim=0).view(-1, num_stocks)
        actuals_l = torch.cat(all_actuals_l, dim=0).view(-1, num_stocks)

        mse_per_stock_h = mse_criterion(preds_h, actuals_h).mean(dim=0)
        qlike_per_stock_h = qlike_criterion(preds_h, actuals_h).mean(dim=0)
        avg_mse_h = mse_per_stock_h.mean().item()
        avg_qlike_h = qlike_per_stock_h.mean().item()

        avg_mse_l = np.nan
        avg_qlike_l = np.nan
        num_samples = preds_l.shape[0]
        if num_samples > 0 and num_samples % intraday_points == 0:
            num_days = num_samples // intraday_points
            avg_preds_per_day = preds_l.reshape(
                num_days, intraday_points, num_stocks
            ).mean(dim=1)
            actuals_per_day = actuals_l.reshape(
                num_days, intraday_points, num_stocks
            )[:, -1, :]
            mse_per_stock_l = mse_criterion(avg_preds_per_day, actuals_per_day).mean(dim=0)
            qlike_per_stock_l = qlike_criterion(avg_preds_per_day, actuals_per_day).mean(dim=0)
            avg_mse_l = mse_per_stock_l.mean().item()
            avg_qlike_l = qlike_per_stock_l.mean().item()

        rmse_h = math.sqrt(avg_mse_h) if not np.isnan(avg_mse_h) else np.nan
        rmse_l = math.sqrt(avg_mse_l) if not np.isnan(avg_mse_l) else np.nan
        return {
            "mse_h": avg_mse_h,
            "rmse_h": rmse_h,
            "qlike_h": avg_qlike_h,
            "mse_l": avg_mse_l,
            "rmse_l": rmse_l,
            "qlike_l": avg_qlike_l,
        }

    validation_metrics = evaluate_on_set(validation_loader, validation_dataset)
    test_metrics = evaluate_on_set(test_loader, test_dataset)

    return {
        "min_validation_loss": min_validation_loss,
        "best_epoch": best_epoch,
        **{f"validation_{k}": v for k, v in validation_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
