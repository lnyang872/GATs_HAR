from copy import deepcopy


DEFAULT_NODE_FEATURE_FLAGS = {
    "har": True,
    "self_volatility": True,
    "covolatility": True,
    "correlation": True,
}


DEFAULT_EDGE_FEATURE_FLAGS = {
    "covolatility": True,
    "source_volatility": True,
    "target_volatility": True,
}


NODE_FEATURE_GROUP_ORDER = (
    "har",
    "self_volatility",
    "covolatility",
    "correlation",
)


EDGE_FEATURE_GROUP_ORDER = (
    "covolatility",
    "source_volatility",
    "target_volatility",
)


VALID_TRAINING_OBJECTIVES = {"joint", "high_only", "low_only"}


def _merge_bool_flags(defaults: dict, overrides: dict | None) -> dict:
    flags = deepcopy(defaults)
    if overrides:
        for key in defaults:
            if key in overrides:
                flags[key] = bool(overrides[key])
    return flags


def get_full_node_feature_flags() -> dict:
    return deepcopy(DEFAULT_NODE_FEATURE_FLAGS)


def get_full_edge_feature_flags() -> dict:
    return deepcopy(DEFAULT_EDGE_FEATURE_FLAGS)


def get_node_feature_flags(config: dict) -> dict:
    flags = _merge_bool_flags(DEFAULT_NODE_FEATURE_FLAGS, config.get("node_feature_flags"))
    if "use_har_features" in config:
        flags["har"] = bool(config["use_har_features"])
    if not bool(config.get("include_energy", True)):
        flags["correlation"] = False
    return flags


def get_edge_feature_flags(config: dict) -> dict:
    return _merge_bool_flags(DEFAULT_EDGE_FEATURE_FLAGS, config.get("edge_feature_flags"))


def sync_feature_flags(config: dict) -> tuple[dict, dict]:
    node_flags = get_node_feature_flags(config)
    edge_flags = get_edge_feature_flags(config)
    config["use_har_features"] = node_flags["har"]
    config["node_feature_flags"] = node_flags
    config["edge_feature_flags"] = edge_flags
    return node_flags, edge_flags


def get_training_objective(config: dict) -> str:
    objective = config.get("training_objective", "low_only")
    if objective not in VALID_TRAINING_OBJECTIVES:
        raise ValueError(
            f"Unsupported training_objective: {objective}. "
            f"Expected one of {sorted(VALID_TRAINING_OBJECTIVES)}."
        )
    return objective


def _get_group_widths(num_stocks: int, num_energy: int) -> dict:
    return {
        "stock": {
            "har": 3,
            "self_volatility": 1,
            "covolatility": max(num_stocks - 1, 0),
            "correlation": max(num_energy, 0),
        },
        "energy": {
            "har": 3,
            "self_volatility": 1,
            "covolatility": max(num_stocks, 0),
            "correlation": max(num_energy - 1, 0),
        },
        "edge": {
            "covolatility": 1,
            "source_volatility": 1,
            "target_volatility": 1,
        },
    }


def get_feature_dimensions(
    *,
    num_stocks: int,
    num_energy: int,
    seq_length: int,
    include_energy: bool,
    node_flags: dict,
    edge_flags: dict,
) -> dict:
    group_widths = _get_group_widths(num_stocks, num_energy)

    stock_feature_dim_per_step = 0
    for group_name in NODE_FEATURE_GROUP_ORDER:
        if group_name == "correlation" and not include_energy:
            continue
        if node_flags[group_name]:
            stock_feature_dim_per_step += group_widths["stock"][group_name]

    energy_feature_dim_per_step = 0
    if include_energy:
        for group_name in NODE_FEATURE_GROUP_ORDER:
            if node_flags[group_name]:
                energy_feature_dim_per_step += group_widths["energy"][group_name]

    edge_feature_dim_per_step = 0
    for group_name in EDGE_FEATURE_GROUP_ORDER:
        if edge_flags[group_name]:
            edge_feature_dim_per_step += group_widths["edge"][group_name]

    return {
        "stock_feature_dim_per_step": stock_feature_dim_per_step,
        "energy_feature_dim_per_step": energy_feature_dim_per_step,
        "edge_feature_dim_per_step": edge_feature_dim_per_step,
        "stock_feature_dim": stock_feature_dim_per_step * seq_length,
        "energy_feature_dim": energy_feature_dim_per_step * seq_length,
        "edge_feature_dim": edge_feature_dim_per_step * seq_length,
    }


def get_full_feature_dimensions(
    *,
    num_stocks: int,
    num_energy: int,
    seq_length: int,
) -> dict:
    return get_feature_dimensions(
        num_stocks=num_stocks,
        num_energy=num_energy,
        seq_length=seq_length,
        include_energy=True,
        node_flags=get_full_node_feature_flags(),
        edge_flags=get_full_edge_feature_flags(),
    )


def validate_feature_dimensions(dimensions: dict, include_energy: bool) -> None:
    if dimensions["stock_feature_dim_per_step"] <= 0:
        raise ValueError("At least one stock node feature group must remain enabled.")
    if include_energy and dimensions["energy_feature_dim_per_step"] <= 0:
        raise ValueError("At least one energy node feature group must remain enabled.")


def _build_column_indices(group_widths: dict, active_flags: dict, seq_length: int) -> list[int]:
    per_step_width = sum(group_widths.values())
    offsets = {}
    offset = 0
    for group_name, width in group_widths.items():
        offsets[group_name] = (offset, offset + width)
        offset += width

    column_indices = []
    for step_idx in range(seq_length):
        step_base = step_idx * per_step_width
        for group_name in group_widths:
            if active_flags.get(group_name, False):
                start, end = offsets[group_name]
                column_indices.extend(range(step_base + start, step_base + end))
    return column_indices


def get_runtime_feature_indices(
    *,
    num_stocks: int,
    num_energy: int,
    seq_length: int,
    include_energy: bool,
    node_flags: dict,
    edge_flags: dict,
) -> dict:
    group_widths = _get_group_widths(num_stocks, num_energy)

    stock_flags = deepcopy(node_flags)
    if not include_energy:
        stock_flags["correlation"] = False

    return {
        "stock": _build_column_indices(group_widths["stock"], stock_flags, seq_length),
        "energy": (
            _build_column_indices(group_widths["energy"], node_flags, seq_length)
            if include_energy
            else []
        ),
        "edge": _build_column_indices(group_widths["edge"], edge_flags, seq_length),
    }


def build_full_dataset_cache_name(config: dict) -> str:
    seq_length = config["seq_length"]
    intraday_points = int(config.get("intraday_points", 3))

    if intraday_points == 3:
        return f"final_hetero_dataset_seq_{seq_length}"
    return f"final_hetero_dataset_seq_{seq_length}_ip_{intraday_points}_full"


def build_dataset_cache_name(config: dict) -> str:
    return build_full_dataset_cache_name(config)
