import argparse
from pathlib import Path

import numpy as np
import yaml

from economic_eval_common import nanmean, summarize_groups, to_sigma, write_csv
from high_freq_eval_base import HighFreqEvalBase


class HighFreqHedgingRunner(HighFreqEvalBase):
    def __init__(self, config, config_path):
        super(HighFreqHedgingRunner, self).__init__(config, config_path)
        self.pairing_cfg = config["pairing"]
        self.hedging_cfg = config["hedging"]
        self.corr_window = int(self.pairing_cfg.get("corr_window", 22 * self.intraday_points))
        self.sigma_window = int(self.hedging_cfg.get("sigma_window", self.corr_window))

    def _clip_value(self, value, lower, upper):
        if lower is not None:
            value = max(value, float(lower))
        if upper is not None:
            value = min(value, float(upper))
        return value

    def _build_corr_series_from_returns(self, stock_returns, future_returns):
        values = []
        aligned_stock = []
        aligned_future = []

        for stock_value, future_value in zip(stock_returns, future_returns):
            if np.isnan(stock_value) or np.isnan(future_value):
                values.append(float("nan"))
                continue

            aligned_stock.append(float(stock_value))
            aligned_future.append(float(future_value))

            left = max(0, len(aligned_stock) - self.corr_window)
            window_stock = np.asarray(aligned_stock[left:], dtype=float)
            window_future = np.asarray(aligned_future[left:], dtype=float)
            if (
                window_stock.size < 2
                or np.std(window_stock) <= 0.0
                or np.std(window_future) <= 0.0
            ):
                values.append(float("nan"))
                continue
            values.append(float(np.corrcoef(window_stock, window_future)[0, 1]))

        return values

    def _build_rolling_sigma_series(self, return_values, window_size):
        values = []
        aligned = []

        for return_value in return_values:
            if np.isnan(return_value):
                values.append(float("nan"))
                continue

            aligned.append(float(return_value))
            left = max(0, len(aligned) - window_size)
            window = np.asarray(aligned[left:], dtype=float)
            if window.size < 2 or np.std(window, ddof=1) <= 0.0:
                values.append(float("nan"))
                continue
            values.append(float(np.std(window, ddof=1)))

        return values

    def _select_futures(self, stock_name, start_index, window_length):
        pairing_mode = self.pairing_cfg.get("mode", "manual")
        if pairing_mode == "manual":
            manual_map = self.pairing_cfg.get("manual_map", {})
            if stock_name not in manual_map:
                raise ValueError("pairing.manual_map is missing stock '{0}'.".format(stock_name))
            configured = manual_map[stock_name]
            future_names = [configured] if isinstance(configured, str) else list(configured)
            return [(future_name, None) for future_name in future_names]

        if pairing_mode == "all_futures":
            return [(future_name, None) for future_name in self.future_names]

        best_future = None
        best_score = -1.0
        stock_series = self._load_stock_series(stock_name)
        full_stock_returns = stock_series["returns"]
        for future_name in self.future_names:
            future_series = self._load_future_series(future_name)
            full_future_returns = future_series["returns"]
            corr_values = self._build_corr_series_from_returns(full_stock_returns, full_future_returns)
            aligned = [
                value
                for value in corr_values[start_index : start_index + window_length]
                if not np.isnan(value)
            ]
            if not aligned:
                continue
            score = float(np.mean(np.abs(np.asarray(aligned, dtype=float))))
            if score > best_score:
                best_future = future_name
                best_score = score
        if best_future is None:
            raise ValueError("Could not auto-select a future for stock '{0}'.".format(stock_name))
        return [(best_future, None)]

    def _compute_hedging_detail(self, model_name, stock_name, future_name, start_index, predicted_values):
        lag_steps = int(self.hedging_cfg.get("lag_steps", 1))
        beta_min = self.hedging_cfg.get("beta_clip_min")
        beta_max = self.hedging_cfg.get("beta_clip_max")

        stock_series = self._load_stock_series(stock_name)
        future_series = self._load_future_series(future_name)
        full_stock_returns = stock_series["returns"]
        full_future_returns = future_series["returns"]
        stock_window = full_stock_returns[start_index : start_index + len(predicted_values)]
        future_window = full_future_returns[start_index : start_index + len(predicted_values)]
        corr_series = self._build_corr_series_from_returns(full_stock_returns, full_future_returns)
        future_sigma_series = self._build_rolling_sigma_series(full_future_returns, self.sigma_window)

        stock_return_values = []
        hedged_return_values = []
        beta_values = []
        used_corr_values = []
        used_step_labels = []

        for local_idx, predicted_value in enumerate(predicted_values):
            global_idx = start_index + local_idx
            previous_global_idx = global_idx - lag_steps
            if previous_global_idx < 0:
                continue
            stock_return = stock_window[local_idx]
            future_return = future_window[local_idx]
            corr_value = corr_series[previous_global_idx]
            future_sigma_value = future_sigma_series[previous_global_idx]
            if np.isnan(stock_return) or np.isnan(future_return) or np.isnan(corr_value):
                continue

            predicted_sigma = to_sigma(
                predicted_value,
                self.prediction_value_type,
                self.variance_epsilon,
            )
            if not np.isfinite(future_sigma_value) or future_sigma_value <= 1e-8:
                beta_value = 0.0
            else:
                beta_value = corr_value * predicted_sigma / future_sigma_value
            beta_value = self._clip_value(beta_value, beta_min, beta_max)

            stock_return_values.append(float(stock_return))
            hedged_return_values.append(float(stock_return) - beta_value * float(future_return))
            beta_values.append(beta_value)
            used_corr_values.append(float(corr_value))
            used_step_labels.append(self._format_step_label(stock_series, start_index + local_idx))

        if len(stock_return_values) < 2:
            stock_return_variance = float("nan")
            hedged_return_variance = float("nan")
            hedging_effectiveness = float("nan")
        else:
            stock_return_variance = float(np.var(stock_return_values, ddof=1))
            hedged_return_variance = float(np.var(hedged_return_values, ddof=1))
            hedging_effectiveness = 1.0 - (hedged_return_variance / stock_return_variance)

        return {
            "model": model_name,
            "stock": stock_name,
            "future": future_name,
            "num_observations": len(stock_return_values),
            "window_start_step": used_step_labels[0] if used_step_labels else "",
            "window_end_step": used_step_labels[-1] if used_step_labels else "",
            "mean_beta": nanmean(beta_values),
            "mean_abs_beta": nanmean(np.abs(np.asarray(beta_values, dtype=float))) if beta_values else float("nan"),
            "mean_corr": nanmean(used_corr_values),
            "mean_abs_corr": nanmean(np.abs(np.asarray(used_corr_values, dtype=float))) if used_corr_values else float("nan"),
            "stock_return_variance": stock_return_variance,
            "hedged_return_variance": hedged_return_variance,
            "hedging_effectiveness": hedging_effectiveness,
        }

    def validate_inputs(self):
        print("--- Checking high-frequency HE inputs")
        self.validate_common_inputs(need_future=True)

    def run(self):
        self.validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        hedging_rows = []
        pairing_rows = []
        alignment_rows = []

        for model_name, model_cfg in self.models_cfg.items():
            workbook_path = str(Path(model_cfg["prediction_workbook"]).resolve())
            predictions = self._load_prediction_workbook(workbook_path)
            print("--- Running high-frequency HE for model: {0}".format(model_name))

            for stock_name, prediction_data in self.iter_sorted_prediction_items(predictions):
                predicted_values = list(prediction_data["predicted"])
                stock_series = self._load_stock_series(stock_name)
                window_info = self._resolve_model_evaluation_window(
                    model_name,
                    len(stock_series["variances"]),
                    len(predicted_values),
                )
                start_index = window_info["window_start_index"]

                selected_futures = self._select_futures(
                    stock_name,
                    start_index,
                    len(predicted_values),
                )

                for future_name, _ in selected_futures:
                    alignment_rows.append(
                        {
                            "model": model_name,
                            "stock": stock_name,
                            "future": future_name,
                            "prediction_workbook": workbook_path,
                            "sample_window_start_index": window_info["window_start_sample_index"],
                            "sample_window_end_index": window_info["window_end_sample_index"],
                            "window_start_index": start_index,
                            "window_end_index": start_index + len(predicted_values) - 1,
                            "window_start_step": self._format_step_label(stock_series, start_index),
                            "window_end_step": self._format_step_label(stock_series, start_index + len(predicted_values) - 1),
                            "prediction_rows": len(predicted_values),
                            "dataset_labels": "|".join(sorted(set(prediction_data["dataset"]))),
                            "aligned_dataset_labels": window_info["selected_labels"],
                        }
                    )
                    pairing_rows.append(
                        {
                            "model": model_name,
                            "stock": stock_name,
                            "future": future_name,
                            "pairing_mode": self.pairing_cfg.get("mode", "manual"),
                        }
                    )
                    hedging_rows.append(
                        self._compute_hedging_detail(
                            model_name,
                            stock_name,
                            future_name,
                            start_index,
                            predicted_values,
                        )
                    )

        hedging_summary = summarize_groups(
            hedging_rows,
            group_keys=["model"],
            numeric_keys=[
                "num_observations",
                "mean_beta",
                "mean_abs_beta",
                "mean_corr",
                "mean_abs_corr",
                "hedging_effectiveness",
            ],
        )

        write_csv(self.output_dir / "hedging_detail.csv", hedging_rows)
        write_csv(self.output_dir / "hedging_summary.csv", hedging_summary)
        write_csv(self.output_dir / "selected_pairs.csv", pairing_rows)
        if self.output_cfg.get("write_alignment_detail", True):
            write_csv(self.output_dir / "alignment_detail.csv", alignment_rows)

        print("--- Finished high-frequency HE")


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required_sections = ["data", "evaluation", "pairing", "hedging", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError("Config is missing section '{0}'.".format(section))
    if not config.get("models") and not config.get("prediction_sources"):
        raise ValueError("Config must provide either 'models' or 'prediction_sources'.")
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run high-frequency hedging effectiveness tests.")
    parser.add_argument("--config", default="config/high_freq_he.yaml", help="Path to YAML config file.")
    parser.add_argument("--check-only", action="store_true", help="Only validate inputs and config, do not run the tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    runner = HighFreqHedgingRunner(config, config_path)
    if args.check_only:
        runner.validate_inputs()
        print("--- Check completed")
        return
    runner.run()


if __name__ == "__main__":
    main()
