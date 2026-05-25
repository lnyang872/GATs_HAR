import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_HDF5_FILES = (
    "processed_data1001/vol_covol.h5",
    "processed_data1001/volvol_covolvol.h5",
)
DEFAULT_NODE_INFO_FILE = "node_info.json"
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split HDF5 matrices into train/validation/test by time order, "
            "standardize each split with its own statistics, and export stats."
        )
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root folder containing processed_data1001 and node_info.json. "
            "If omitted, the script will auto-detect it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output folder for standardized files and statistics. "
            "Default: <data-root>/processed_data1001/split_standardized"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def detect_data_root(explicit_root: Optional[str]) -> Path:
    if explicit_root is not None:
        root = Path(explicit_root).resolve()
        validate_data_root(root)
        return root

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd().resolve(),
        script_dir,
        script_dir.parent,
    ]

    for candidate in candidates:
        if is_valid_data_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not detect data root automatically. "
        "Please pass --data-root explicitly."
    )


def is_valid_data_root(root: Path) -> bool:
    return all((root / rel_path).exists() for rel_path in DEFAULT_HDF5_FILES) and (
        root / DEFAULT_NODE_INFO_FILE
    ).exists()


def validate_data_root(root: Path) -> None:
    missing = [
        str(root / rel_path)
        for rel_path in (*DEFAULT_HDF5_FILES, DEFAULT_NODE_INFO_FILE)
        if not (root / rel_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(missing)
        )


def load_node_info(node_info_path: Path) -> dict:
    with node_info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_indices(node_info: dict) -> tuple[list[int], list[int]]:
    node_order = node_info["node_order"]
    stock_indices = [node_order.index(node_id) for node_id in node_info["stock_ids"]]
    energy_indices = [node_order.index(node_id) for node_id in node_info["energy_ids"]]
    return stock_indices, energy_indices


def sort_h5_keys(keys: list[str]) -> list[str]:
    def sort_key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(keys, key=sort_key)


def build_split_map(
    keys: list[str],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.6f}."
        )

    total = len(keys)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    split_keys = {
        "train": keys[:train_end],
        "validation": keys[train_end:validation_end],
        "test": keys[validation_end:],
    }

    for split_name, split_key_list in split_keys.items():
        if not split_key_list:
            raise ValueError(f"Split '{split_name}' is empty. Adjust ratios or data size.")

    return split_keys


def _extract_groups(
    matrix: np.ndarray,
    stock_indices: list[int],
    energy_indices: list[int],
) -> dict[str, np.ndarray]:
    stock_diag = matrix[stock_indices, stock_indices].astype(np.float64)
    energy_diag = matrix[energy_indices, energy_indices].astype(np.float64)

    ss_block = matrix[np.ix_(stock_indices, stock_indices)].astype(np.float64)
    ee_block = matrix[np.ix_(energy_indices, energy_indices)].astype(np.float64)
    se_block = matrix[np.ix_(stock_indices, energy_indices)].astype(np.float64)

    return {
        "stock_diag": stock_diag,
        "energy_diag": energy_diag,
        "ss_off_diag": ss_block[np.triu_indices_from(ss_block, k=1)],
        "ee_off_diag": ee_block[np.triu_indices_from(ee_block, k=1)],
        "se_off_diag": se_block.flatten(),
    }


def _summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_split_stats(
    file_path: Path,
    split_keys: dict,
    stock_indices: list[int],
    energy_indices: list[int],
) -> dict:
    split_stats = {}
    with h5py.File(file_path, "r") as h5_file:
        for split_name, keys_for_split in split_keys.items():
            grouped_values = {
                "stock_diag": [],
                "energy_diag": [],
                "ss_off_diag": [],
                "ee_off_diag": [],
                "se_off_diag": [],
            }

            for key in tqdm(
                keys_for_split,
                desc=f"Collecting stats [{file_path.name} | {split_name}]",
            ):
                matrix = h5_file[key][()]
                extracted = _extract_groups(matrix, stock_indices, energy_indices)
                for group_name, values in extracted.items():
                    grouped_values[group_name].append(values)

            split_stats[split_name] = {
                group_name: _summarize(np.concatenate(value_list))
                for group_name, value_list in grouped_values.items()
            }

    return split_stats


def safe_standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    if std <= EPS:
        return np.zeros_like(values, dtype=np.float64)
    return (values.astype(np.float64) - mean) / std


def standardize_matrix(
    matrix: np.ndarray,
    stats: dict,
    stock_indices: list[int],
    energy_indices: list[int],
) -> np.ndarray:
    standardized = np.zeros_like(matrix, dtype=np.float64)

    if stock_indices:
        ss_block = matrix[np.ix_(stock_indices, stock_indices)]
        ss_stats = stats["ss_off_diag"]
        standardized[np.ix_(stock_indices, stock_indices)] = safe_standardize(
            ss_block, ss_stats["mean"], ss_stats["std"]
        )

        stock_diag = matrix[stock_indices, stock_indices]
        stock_diag_stats = stats["stock_diag"]
        standardized[stock_indices, stock_indices] = safe_standardize(
            stock_diag, stock_diag_stats["mean"], stock_diag_stats["std"]
        )

    if energy_indices:
        ee_block = matrix[np.ix_(energy_indices, energy_indices)]
        ee_stats = stats["ee_off_diag"]
        standardized[np.ix_(energy_indices, energy_indices)] = safe_standardize(
            ee_block, ee_stats["mean"], ee_stats["std"]
        )

        energy_diag = matrix[energy_indices, energy_indices]
        energy_diag_stats = stats["energy_diag"]
        standardized[energy_indices, energy_indices] = safe_standardize(
            energy_diag, energy_diag_stats["mean"], energy_diag_stats["std"]
        )

    if stock_indices and energy_indices:
        se_block = matrix[np.ix_(stock_indices, energy_indices)]
        se_stats = stats["se_off_diag"]
        se_block_std = safe_standardize(se_block, se_stats["mean"], se_stats["std"])
        standardized[np.ix_(stock_indices, energy_indices)] = se_block_std
        standardized[np.ix_(energy_indices, stock_indices)] = se_block_std.T

    return standardized


def build_output_paths(output_dir: Path, source_file: Path) -> dict[str, Path]:
    stem = source_file.stem
    return {
        "full_standardized": output_dir / f"{stem}_split_standardized.h5",
        "train_standardized": output_dir / f"{stem}_train_standardized.h5",
        "validation_standardized": output_dir / f"{stem}_validation_standardized.h5",
        "test_standardized": output_dir / f"{stem}_test_standardized.h5",
        "stats_json": output_dir / f"{stem}_split_stats.json",
        "stats_csv": output_dir / f"{stem}_split_stats.csv",
    }


def write_stats_outputs(
    output_paths: dict[str, Path],
    source_file: Path,
    split_keys: dict,
    split_stats: dict,
    split_ratios: dict,
) -> None:
    stats_payload = {
        "source_file": str(source_file),
        "split_ratios": split_ratios,
        "total_keys": sum(len(keys) for keys in split_keys.values()),
        "splits": {},
    }

    for split_name, keys_for_split in split_keys.items():
        stats_payload["splits"][split_name] = {
            "num_keys": len(keys_for_split),
            "first_key": keys_for_split[0],
            "last_key": keys_for_split[-1],
            "stats": split_stats[split_name],
        }

    with output_paths["stats_json"].open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=4)

    with output_paths["stats_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "split",
                "num_keys",
                "first_key",
                "last_key",
                "group",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ],
        )
        writer.writeheader()

        for split_name, keys_for_split in split_keys.items():
            for group_name, stats in split_stats[split_name].items():
                writer.writerow(
                    {
                        "source_file": str(source_file),
                        "split": split_name,
                        "num_keys": len(keys_for_split),
                        "first_key": keys_for_split[0],
                        "last_key": keys_for_split[-1],
                        "group": group_name,
                        **stats,
                    }
                )


def standardize_file_by_split(
    file_path: Path,
    output_dir: Path,
    split_keys: dict,
    split_stats: dict,
    stock_indices: list[int],
    energy_indices: list[int],
) -> dict[str, Path]:
    output_paths = build_output_paths(output_dir, file_path)

    split_output_handles = {}
    try:
        with h5py.File(file_path, "r") as input_h5, h5py.File(
            output_paths["full_standardized"], "w"
        ) as full_output_h5:
            for split_name in ("train", "validation", "test"):
                split_output_handles[split_name] = h5py.File(
                    output_paths[f"{split_name}_standardized"], "w"
                )

            for split_name, keys_for_split in split_keys.items():
                split_output_h5 = split_output_handles[split_name]
                stats = split_stats[split_name]

                for key in tqdm(
                    keys_for_split,
                    desc=f"Standardizing [{file_path.name} | {split_name}]",
                ):
                    matrix = input_h5[key][()]
                    standardized_matrix = standardize_matrix(
                        matrix,
                        stats,
                        stock_indices,
                        energy_indices,
                    )

                    full_output_h5.create_dataset(key, data=standardized_matrix)
                    split_output_h5.create_dataset(key, data=standardized_matrix)
    finally:
        for handle in split_output_handles.values():
            handle.close()

    return output_paths


def main():
    args = parse_args()
    data_root = detect_data_root(args.data_root)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (data_root / "processed_data1001" / "split_standardized")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    node_info = load_node_info(data_root / DEFAULT_NODE_INFO_FILE)
    stock_indices, energy_indices = build_indices(node_info)
    split_ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }

    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    for relative_file_path in DEFAULT_HDF5_FILES:
        source_file = data_root / relative_file_path
        print("\n" + "=" * 80)
        print(f"Processing: {source_file.name}")
        print("=" * 80)

        with h5py.File(source_file, "r") as h5_file:
            keys = sort_h5_keys(list(h5_file.keys()))

        split_keys = build_split_map(
            keys,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )

        print(
            "Split sizes: "
            f"train={len(split_keys['train'])}, "
            f"validation={len(split_keys['validation'])}, "
            f"test={len(split_keys['test'])}"
        )

        split_stats = compute_split_stats(
            source_file,
            split_keys,
            stock_indices,
            energy_indices,
        )
        output_paths = standardize_file_by_split(
            source_file,
            output_dir,
            split_keys,
            split_stats,
            stock_indices,
            energy_indices,
        )
        write_stats_outputs(
            output_paths,
            source_file,
            split_keys,
            split_stats,
            split_ratios,
        )

        print("Saved outputs:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")

    print("\nFinished.")


if __name__ == "__main__":
    main()
