import copy
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm

from utils.feature_config import (
    get_edge_feature_flags,
    get_feature_dimensions,
    get_full_feature_dimensions,
    get_full_node_feature_flags,
    get_node_feature_flags,
    get_runtime_feature_indices,
    validate_feature_dimensions,
)


class FinalHeteroDataset(InMemoryDataset):
    """
    Final heterogeneous graph dataset used by the GATs-HAR pipeline.

    Cache policy:
    - processed cache always stores full node/edge features with energy relations.
    - ablation settings are applied at runtime via column slicing in `get()`.
    """

    def __init__(
        self,
        hdf5_file_vol,
        hdf5_file_volvol,
        stock_har_rv_folder,
        energy_har_rv_folder,
        stock_energy_corr_folder,
        energy_energy_corr_folder,
        node_info_file,
        root,
        transform=None,
        pre_transform=None,
        seq_length=15,
        intraday_points=3,
        include_energy=True,
        use_har_features=True,
        node_feature_flags=None,
        edge_feature_flags=None,
    ):
        self.hdf5_file_vol = hdf5_file_vol
        self.hdf5_file_volvol = hdf5_file_volvol
        self.stock_har_rv_folder = stock_har_rv_folder
        self.energy_har_rv_folder = energy_har_rv_folder
        self.stock_energy_corr_folder = stock_energy_corr_folder
        self.energy_energy_corr_folder = energy_energy_corr_folder
        self.node_info_file = node_info_file
        self.seq_length = seq_length
        self.intraday_points = intraday_points
        self.include_energy = include_energy

        self.node_feature_flags = get_node_feature_flags(
            {
                "include_energy": include_energy,
                "use_har_features": use_har_features,
                "node_feature_flags": node_feature_flags,
            }
        )
        self.edge_feature_flags = get_edge_feature_flags(
            {"edge_feature_flags": edge_feature_flags}
        )
        self.full_node_feature_flags = get_full_node_feature_flags()

        with open(node_info_file, "r", encoding="utf-8") as f:
            self.node_info = json.load(f)

        self.stock_ids = self.node_info["stock_ids"]
        self.energy_ids = self.node_info["energy_ids"]
        self.node_order = self.node_info["node_order"]

        self.num_stocks = len(self.stock_ids)
        self.num_energy = len(self.energy_ids)
        self.num_nodes = len(self.node_order)

        self.node_to_idx = {name: i for i, name in enumerate(self.node_order)}
        self.stock_local_index = {
            node_id: idx for idx, node_id in enumerate(self.stock_ids)
        }

        self.stock_indices = torch.tensor(
            [self.node_to_idx[sid] for sid in self.stock_ids], dtype=torch.long
        )
        self.energy_indices = torch.tensor(
            [self.node_to_idx[eid] for eid in self.energy_ids], dtype=torch.long
        )

        self.feature_dimensions = get_feature_dimensions(
            num_stocks=self.num_stocks,
            num_energy=self.num_energy if self.include_energy else 0,
            seq_length=self.seq_length,
            include_energy=self.include_energy,
            node_flags=self.node_feature_flags,
            edge_flags=self.edge_feature_flags,
        )
        validate_feature_dimensions(self.feature_dimensions, self.include_energy)

        self.full_feature_dimensions = get_full_feature_dimensions(
            num_stocks=self.num_stocks,
            num_energy=self.num_energy,
            seq_length=self.seq_length,
        )
        self.runtime_feature_indices = get_runtime_feature_indices(
            num_stocks=self.num_stocks,
            num_energy=self.num_energy,
            seq_length=self.seq_length,
            include_energy=self.include_energy,
            node_flags=self.node_feature_flags,
            edge_flags=self.edge_feature_flags,
        )

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def _load_features_to_dict(self, folder_path, file_ids):
        if folder_path is None or not os.path.exists(folder_path):
            return {}

        feature_dict = {}
        for asset_id in file_ids:
            try:
                file_path = next(
                    os.path.join(folder_path, file_name)
                    for file_name in os.listdir(folder_path)
                    if file_name.startswith(asset_id)
                )
            except StopIteration:
                print(
                    f"Warning: feature file for asset {asset_id} "
                    f"was not found in {folder_path}."
                )
                continue

            feature_dict[asset_id] = pd.read_csv(
                file_path, header=None, index_col=0, parse_dates=True
            )
        return feature_dict

    @staticmethod
    def _get_corr_value(all_corr_df, left_id, right_id, current_date):
        if all_corr_df.empty:
            return 0.0
        pair_1 = f"{left_id}_{right_id}"
        pair_2 = f"{right_id}_{left_id}"
        if pair_1 in all_corr_df.columns:
            return float(all_corr_df.at[current_date, pair_1])
        if pair_2 in all_corr_df.columns:
            return float(all_corr_df.at[current_date, pair_2])
        return 0.0

    def _build_stock_features(
        self,
        node_idx,
        node_id,
        current_date,
        cov_matrix,
        all_har_features,
        all_corr_df,
    ):
        components = []

        if self.full_node_feature_flags["har"]:
            components.append(all_har_features[node_id].loc[current_date].values)

        if self.full_node_feature_flags["self_volatility"]:
            components.append(np.array([cov_matrix[node_idx, node_idx]], dtype=np.float32))

        if self.full_node_feature_flags["covolatility"]:
            stock_local_idx = self.stock_local_index[node_id]
            stock_covols = np.delete(cov_matrix[node_idx, self.stock_indices], stock_local_idx)
            components.append(np.asarray(stock_covols, dtype=np.float32))

        if self.full_node_feature_flags["correlation"]:
            corr_values = [
                self._get_corr_value(all_corr_df, node_id, energy_id, current_date)
                for energy_id in self.energy_ids
            ]
            components.append(np.asarray(corr_values, dtype=np.float32))

        return np.concatenate(components).astype(np.float32)

    def _build_energy_features(
        self,
        node_idx,
        node_id,
        current_date,
        cov_matrix,
        all_har_features,
        all_corr_df,
    ):
        components = []

        if self.full_node_feature_flags["har"]:
            components.append(all_har_features[node_id].loc[current_date].values)

        if self.full_node_feature_flags["self_volatility"]:
            components.append(np.array([cov_matrix[node_idx, node_idx]], dtype=np.float32))

        if self.full_node_feature_flags["covolatility"]:
            components.append(np.asarray(cov_matrix[node_idx, self.stock_indices], dtype=np.float32))

        if self.full_node_feature_flags["correlation"]:
            corr_values = []
            for other_energy_id in self.energy_ids:
                if node_id == other_energy_id:
                    continue
                corr_values.append(
                    self._get_corr_value(all_corr_df, node_id, other_energy_id, current_date)
                )
            components.append(np.asarray(corr_values, dtype=np.float32))

        return np.concatenate(components).astype(np.float32)

    @staticmethod
    def _build_global_edge_attr(volvol_matrix, global_edge_index):
        if global_edge_index.numel() == 0:
            return None

        diagonal = torch.tensor(np.diag(volvol_matrix), dtype=torch.float32)
        adj_matrix = volvol_matrix.copy()
        np.fill_diagonal(adj_matrix, 0)
        covars = torch.tensor(
            adj_matrix[global_edge_index[0], global_edge_index[1]],
            dtype=torch.float32,
        )

        return torch.stack(
            [
                covars,
                diagonal[global_edge_index[0]],
                diagonal[global_edge_index[1]],
            ],
            dim=1,
        )

    def _apply_runtime_ablation(self, data: HeteroData) -> HeteroData:
        stock_idx = self.runtime_feature_indices["stock"]
        data["stock"].x = data["stock"].x[:, stock_idx].clone()

        if self.include_energy:
            energy_idx = self.runtime_feature_indices["energy"]
            if "energy" in data.node_types:
                data["energy"].x = data["energy"].x[:, energy_idx].clone()
        else:
            if "energy" in data.node_types:
                del data["energy"]
            for rel in [("stock", "to", "energy"), ("energy", "to", "stock")]:
                if rel in data.edge_types:
                    del data[rel]

        edge_idx = self.runtime_feature_indices["edge"]
        for rel in list(data.edge_types):
            if "edge_attr" not in data[rel]:
                continue
            if edge_idx:
                data[rel].edge_attr = data[rel].edge_attr[:, edge_idx].clone()
            else:
                del data[rel]["edge_attr"]

        return data

    def get(self, idx):
        data = super().get(idx)
        data = copy.copy(data)
        return self._apply_runtime_ablation(data)

    def process(self):
        print("\n==========================================================")
        print("====== Building shared final heterogeneous dataset cache ======")
        print(
            f"====== seq_length={self.seq_length}, "
            f"intraday_points={self.intraday_points} ======"
        )
        print("====== cache contents: full node features + full edge features ======")
        print("==========================================================")

        all_har_features = self._load_features_to_dict(
            self.stock_har_rv_folder, self.stock_ids
        )
        all_har_features.update(
            self._load_features_to_dict(
                self.energy_har_rv_folder, self.energy_ids
            )
        )

        if not all_har_features:
            raise FileNotFoundError(
                "No HAR feature files were loaded. Check the configured folders."
            )

        valid_dates = next(iter(all_har_features.values())).index

        corr_df_dict = {}
        corr_folders = [
            self.stock_energy_corr_folder,
            self.energy_energy_corr_folder,
        ]
        for folder in corr_folders:
            if folder is None or not os.path.exists(folder):
                continue
            for pair_file in os.listdir(folder):
                if not pair_file.endswith(".csv"):
                    continue
                pair_name = os.path.splitext(pair_file)[0]
                path = os.path.join(folder, pair_file)
                corr_values = pd.read_csv(path, header=None).values.flatten()
                if len(corr_values) == len(valid_dates):
                    corr_df_dict[pair_name] = pd.Series(corr_values, index=valid_dates)
        all_corr_df = pd.DataFrame(corr_df_dict)

        all_hf_steps = []
        for date in valid_dates:
            day_start_index = valid_dates.get_loc(date) * self.intraday_points
            for point_idx in range(self.intraday_points):
                all_hf_steps.append((day_start_index + point_idx, date))

        relations = [
            ("stock", "to", "stock"),
            ("stock", "to", "energy"),
            ("energy", "to", "stock"),
        ]

        data_list = []
        skipped_samples_count = 0
        iterator_range = len(all_hf_steps) - self.seq_length - self.intraday_points
        edge_dim_per_step = self.full_feature_dimensions["edge_feature_dim_per_step"]

        with h5py.File(self.hdf5_file_vol, "r") as f_vol, h5py.File(
            self.hdf5_file_volvol, "r"
        ) as f_volvol:
            for i in tqdm(range(iterator_range), desc="Creating graph sequences"):
                seq_data_list = []
                is_valid_sequence = True

                for j in range(self.seq_length):
                    hf_global_idx, current_date = all_hf_steps[i + j]

                    try:
                        cov_matrix = np.array(f_vol[str(hf_global_idx)])
                        volvol_matrix = np.array(f_volvol[str(hf_global_idx)])
                    except KeyError:
                        is_valid_sequence = False
                        break

                    data = HeteroData()

                    stock_features_t = []
                    for stock_id in self.stock_ids:
                        stock_node_idx = self.node_to_idx[stock_id]
                        stock_features_t.append(
                            self._build_stock_features(
                                node_idx=stock_node_idx,
                                node_id=stock_id,
                                current_date=current_date,
                                cov_matrix=cov_matrix,
                                all_har_features=all_har_features,
                                all_corr_df=all_corr_df,
                            )
                        )
                    data["stock"].x = torch.tensor(
                        np.stack(stock_features_t), dtype=torch.float32
                    )

                    energy_features_t = []
                    for energy_id in self.energy_ids:
                        energy_node_idx = self.node_to_idx[energy_id]
                        energy_features_t.append(
                            self._build_energy_features(
                                node_idx=energy_node_idx,
                                node_id=energy_id,
                                current_date=current_date,
                                cov_matrix=cov_matrix,
                                all_har_features=all_har_features,
                                all_corr_df=all_corr_df,
                            )
                        )
                    data["energy"].x = torch.tensor(
                        np.stack(energy_features_t), dtype=torch.float32
                    )

                    adj_matrix = volvol_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)
                    global_edge_index = torch.from_numpy(
                        np.vstack(np.where(adj_matrix != 0))
                    ).long()
                    global_edge_attr = self._build_global_edge_attr(
                        volvol_matrix=volvol_matrix, global_edge_index=global_edge_index
                    )

                    for src, rel, dst in relations:
                        if global_edge_index.numel() == 0:
                            continue

                        global_src_indices = (
                            self.stock_indices if src == "stock" else self.energy_indices
                        )
                        global_dst_indices = (
                            self.stock_indices if dst == "stock" else self.energy_indices
                        )

                        mask = torch.isin(
                            global_edge_index[0], global_src_indices
                        ) & torch.isin(global_edge_index[1], global_dst_indices)

                        if mask.sum() == 0:
                            continue

                        src_map = {
                            idx.item(): local_idx
                            for local_idx, idx in enumerate(global_src_indices)
                        }
                        dst_map = {
                            idx.item(): local_idx
                            for local_idx, idx in enumerate(global_dst_indices)
                        }
                        local_edge_index = global_edge_index[:, mask]
                        local_edge_index = torch.stack(
                            [
                                torch.tensor(
                                    [src_map[idx.item()] for idx in local_edge_index[0]],
                                    dtype=torch.long,
                                ),
                                torch.tensor(
                                    [dst_map[idx.item()] for idx in local_edge_index[1]],
                                    dtype=torch.long,
                                ),
                            ],
                            dim=0,
                        )

                        data[src, rel, dst].edge_index = local_edge_index
                        if global_edge_attr is not None:
                            data[src, rel, dst].edge_attr = global_edge_attr[mask]

                    try:
                        next_hf_global_idx, _ = all_hf_steps[i + j + 1]
                        next_cov_matrix = np.array(f_vol[str(next_hf_global_idx)])
                        y_high_all = torch.tensor(
                            np.diag(next_cov_matrix), dtype=torch.float32
                        )
                        data["stock"].y_high = y_high_all[self.stock_indices]

                        next_day_date = all_hf_steps[i + j + self.intraday_points][1]
                        y_low_values = [
                            all_har_features[stock_id].iloc[:, 0].loc[next_day_date]
                            for stock_id in self.stock_ids
                        ]
                        data["stock"].y_low = torch.tensor(
                            y_low_values, dtype=torch.float32
                        )
                    except (KeyError, IndexError):
                        is_valid_sequence = False
                        break

                    seq_data_list.append(data)

                if not is_valid_sequence:
                    continue

                final_data = HeteroData()
                final_data["stock"].x = torch.cat(
                    [step_data["stock"].x for step_data in seq_data_list], dim=1
                )
                final_data["energy"].x = torch.cat(
                    [step_data["energy"].x for step_data in seq_data_list], dim=1
                )

                last_data = seq_data_list[-1]
                for rel in last_data.edge_types:
                    final_data[rel].edge_index = last_data[rel].edge_index

                    if edge_dim_per_step > 0:
                        edge_attr_seq = []
                        num_edges = last_data[rel].edge_index.size(1)
                        for step_data in seq_data_list:
                            if rel in step_data and "edge_attr" in step_data[rel]:
                                edge_attr_seq.append(step_data[rel].edge_attr)
                            else:
                                edge_attr_seq.append(
                                    torch.zeros(
                                        num_edges,
                                        edge_dim_per_step,
                                        dtype=torch.float32,
                                    )
                                )
                        final_data[rel].edge_attr = torch.cat(edge_attr_seq, dim=1)

                final_data["stock"].y_high = last_data["stock"].y_high
                final_data["stock"].y_low = last_data["stock"].y_low

                total_edges = sum(
                    edge_index.size(1)
                    for edge_index in final_data.edge_index_dict.values()
                )
                if total_edges > 0:
                    data_list.append(final_data)
                else:
                    skipped_samples_count += 1

        if skipped_samples_count > 0:
            print(
                f"\n[Dataset Warning] Skipped {skipped_samples_count} samples "
                f"because the final graph had no edges."
            )

        if not data_list:
            raise ValueError("No valid graph samples were created.")

        print("\nSaving processed heterogeneous dataset cache...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dataset cache saved to {self.processed_paths[0]}.")
        print("==========================================================")
