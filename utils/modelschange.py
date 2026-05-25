from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear


class HeteroGNNModel(torch.nn.Module):
    def __init__(
        self,
        stock_feature_dim: int,
        energy_feature_dim: int,
        edge_feature_dim: int,
        heads: int,
        high_freq_output_dim: int,
        low_freq_output_dim: int,
        hidden_layout: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        include_energy: bool = True,
        use_har_features: bool = True,
    ):
        super().__init__()

        if not hidden_layout:
            raise ValueError("hidden_layout must not be empty.")

        self.include_energy = include_energy
        self.use_har_features = use_har_features
        self.edge_feature_dim = edge_feature_dim

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "leaky_relu":
            self.activation_fn = F.leaky_relu
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        print(f"--- [Model Init] include_energy: {self.include_energy}")
        print(f"--- [Model Init] use_har_features: {self.use_har_features}")
        print(
            f"--- [Model Init] input dims -> stock: {stock_feature_dim}, "
            f"energy: {energy_feature_dim}, edge: {edge_feature_dim}"
        )

        first_layer_dim = hidden_layout[0]
        lin_modules = {"stock": Linear(stock_feature_dim, first_layer_dim)}
        if self.include_energy:
            if energy_feature_dim <= 0:
                raise ValueError(
                    "include_energy=True but energy_feature_dim is not positive."
                )
            lin_modules["energy"] = Linear(energy_feature_dim, first_layer_dim)
        self.lin_dict = nn.ModuleDict(lin_modules)

        if self.include_energy:
            self.relations = [
                ("stock", "to", "stock"),
                ("stock", "to", "energy"),
                ("energy", "to", "stock"),
            ]
        else:
            self.relations = [("stock", "to", "stock")]

        print(f"--- [Model Init] relations: {self.relations}")

        self.convs = nn.ModuleList()
        in_channels = first_layer_dim
        for out_channels in hidden_layout:
            conv_dict = {}
            for rel in self.relations:
                conv_kwargs = dict(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    dropout=dropout,
                )
                if self.edge_feature_dim > 0:
                    conv_kwargs["edge_dim"] = self.edge_feature_dim
                conv_dict[rel] = GATConv(**conv_kwargs)

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            in_channels = out_channels * heads

        final_embedding_dim = hidden_layout[-1] * heads
        self.high_freq_head = Linear(final_embedding_dim, 1)
        self.low_freq_head = Linear(final_embedding_dim, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = {
            node_type: self.activation_fn(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.lin_dict
        }

        for conv in self.convs:
            if self.edge_feature_dim > 0:
                current_edge_attr_dict = {
                    key: value
                    for key, value in edge_attr_dict.items()
                    if key in self.relations
                }
                x_dict = conv(
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict=current_edge_attr_dict,
                )
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation_fn(x) for key, x in x_dict.items()}

        if "stock" not in x_dict:
            print("\n" + "=" * 50)
            print("====== [Debug] 'stock' key missing after GNN layers ======")
            print(f"x_dict keys: {list(x_dict.keys())}")
            print(f"edge_index_dict keys: {list(edge_index_dict.keys())}")
            print(f"model relations: {self.relations}")
            print("=" * 50 + "\n")
            return torch.tensor([]), torch.tensor([])

        final_stock_embedding = x_dict["stock"]
        pred_high = self.high_freq_head(final_stock_embedding)
        pred_low = self.low_freq_head(final_stock_embedding)
        return pred_high.squeeze(-1), pred_low.squeeze(-1)
