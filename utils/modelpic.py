# File: utils/modelschange.py

from typing import Dict, List, Optional, Tuple

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
    ):
        super().__init__()

        if not hidden_layout:
            raise ValueError("hidden_layout must not be empty.")

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "leaky_relu":
            self.activation_fn = F.leaky_relu
        elif activation == "tanh":
            self.activation_fn = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        first_layer_dim = hidden_layout[0]
        self.lin_dict = nn.ModuleDict(
            {
                "stock": Linear(stock_feature_dim, first_layer_dim),
                "energy": Linear(energy_feature_dim, first_layer_dim),
            }
        )

        self.convs = nn.ModuleList()
        num_layers = len(hidden_layout)
        in_channels = first_layer_dim

        for i in range(num_layers):
            out_channels = hidden_layout[i]
            conv = HeteroConv(
                {
                    rel: GATConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        heads=heads,
                        concat=True,
                        edge_dim=edge_feature_dim,
                        add_self_loops=False,
                        dropout=dropout,
                    )
                    for rel in [("stock", "to", "stock"), ("stock", "to", "energy"), ("energy", "to", "stock")]
                },
                aggr="sum",
            )
            self.convs.append(conv)
            in_channels = out_channels * heads

        final_embedding_dim = hidden_layout[-1] * heads
        self.high_freq_head = Linear(final_embedding_dim, 1)
        self.low_freq_head = Linear(final_embedding_dim, 1)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Dict[Tuple, torch.Tensor],
        return_attentions: bool = False,
    ):
        x_dict_initial = {node_type: self.activation_fn(self.lin_dict[node_type](x)) for node_type, x in x_dict.items()}

        # Use a temporary variable to propagate node representations through the GNN layers.
        x_dict_processed = x_dict_initial
        for conv in self.convs:
            x_dict_processed = conv(x_dict_processed, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict_processed = {key: self.activation_fn(x) for key, x in x_dict_processed.items()}

        # Make sure the stock embedding exists before creating the outputs.
        if "stock" not in x_dict_processed:
            if return_attentions:
                return torch.empty(0), torch.empty(0), None
            return torch.empty(0), torch.empty(0)

        final_stock_embedding = x_dict_processed["stock"]

        pred_high = self.high_freq_head(final_stock_embedding).squeeze(-1)
        pred_low = self.low_freq_head(final_stock_embedding).squeeze(-1)

        # Optionally perform an additional pass to extract attention weights.
        if return_attentions:
            attention_weights = None
            relation_key = ("stock", "to", "stock")

            # Only attempt this if the first convolution layer and edge relation exist.
            if len(self.convs) > 0 and relation_key in self.convs[0].convs and relation_key in edge_index_dict:
                target_layer = self.convs[0].convs[relation_key]

                # Use the first-layer input representation to compute attention weights.
                _, attention_weights = target_layer(
                    (x_dict_initial["stock"], x_dict_initial["stock"]),
                    edge_index_dict[relation_key],
                    edge_attr=edge_attr_dict.get(relation_key),
                    return_attention_weights=True,
                )

            return pred_high, pred_low, attention_weights

        return pred_high, pred_low
